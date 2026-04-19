// Go2 velocity policy deployment node.
//
// Subscribes: /imu (sensor_msgs/Imu), /joint_states (sensor_msgs/JointState),
//             /cmd_vel (geometry_msgs/Twist)
// Publishes:  /joint_commands (sensor_msgs/JointState) at 500 Hz,
//             /policy_active  (std_msgs/Bool)
//
// Runs an ONNX policy trained by unitree_rl_mjlab on the Unitree-Go2-Flat task.
// The policy runs at 50 Hz; the publisher runs at 500 Hz with zero-order hold
// between inferences (required for mujoco-side PD stability).
//
// Joint data on /joint_states and /joint_commands always includes joint names.
// Consumers use name-based lookups — ordering on the wire does not matter.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnxruntime_cxx_api.h"

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/bool.hpp"

namespace {

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
constexpr int kNumLegMotor = 12;
constexpr int kObsDim = 47;              // 3 + 3 + 3 + 2 + 12 + 12 + 12
constexpr int kActionDim = 12;
constexpr double kPhasePeriod = 0.6;     // seconds, matches training
constexpr double kCmdStandThreshold = 0.1;

// Joint names in the order the policy expects (matching the ONNX training config).
const std::vector<std::string> kJointNames = {
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
};

// Joint name -> internal index (for receiving /joint_states in any order).
const std::unordered_map<std::string, int> kNameToIndex = {
    {"FL_hip_joint", 0}, {"FL_thigh_joint", 1}, {"FL_calf_joint", 2},
    {"FR_hip_joint", 3}, {"FR_thigh_joint", 4}, {"FR_calf_joint", 5},
    {"RL_hip_joint", 6}, {"RL_thigh_joint", 7}, {"RL_calf_joint", 8},
    {"RR_hip_joint", 9}, {"RR_thigh_joint", 10}, {"RR_calf_joint", 11},
};

// Default joint positions (matching kJointNames order and the ONNX training config).
constexpr std::array<float, kNumLegMotor> kDefaultQ = {
    -0.1f, 0.9f, -1.8f,   // FL
     0.1f, 0.9f, -1.8f,   // FR
    -0.1f, 0.9f, -1.8f,   // RL
     0.1f, 0.9f, -1.8f    // RR
};

// -----------------------------------------------------------------------------
// Utility: projected gravity from quaternion (w, x, y, z).
// Rotates world gravity (0, 0, -1) into the body frame via q^{-1}.
// Sanity: q = (1, 0, 0, 0) -> (0, 0, -1).
// -----------------------------------------------------------------------------
inline void projected_gravity(float qw, float qx, float qy, float qz,
                              float out[3]) {
  out[0] = -2.0f * (qx * qz + qw * qy);
  out[1] = 2.0f * (qw * qx - qy * qz);
  out[2] = -1.0f + 2.0f * (qx * qx + qy * qy);
}

}  // namespace

// -----------------------------------------------------------------------------
// ONNX policy wrapper
// -----------------------------------------------------------------------------
class OnnxPolicy {
 public:
  explicit OnnxPolicy(const std::string& model_path)
      : env_(ORT_LOGGING_LEVEL_WARNING, "go2_velocity") {
    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(),
                                              session_options_);

    for (size_t i = 0; i < session_->GetInputCount(); ++i) {
      auto type_info = session_->GetInputTypeInfo(i);
      input_shapes_.push_back(type_info.GetTensorTypeAndShapeInfo().GetShape());
      auto name = session_->GetInputNameAllocated(i, allocator_);
      input_name_strs_.push_back(name.get());
      size_t size = 1;
      for (auto dim : input_shapes_.back()) size *= dim;
      input_sizes_.push_back(size);
    }
    for (auto& s : input_name_strs_) input_names_.push_back(s.c_str());

    auto out_type = session_->GetOutputTypeInfo(0);
    output_shape_ = out_type.GetTensorTypeAndShapeInfo().GetShape();
    auto out_name = session_->GetOutputNameAllocated(0, allocator_);
    output_name_str_ = out_name.get();
    output_name_ = output_name_str_.c_str();
  }

  const std::vector<int64_t>& input_shape(size_t i) const { return input_shapes_[i]; }
  const std::vector<int64_t>& output_shape() const { return output_shape_; }
  size_t input_count() const { return input_names_.size(); }

  // Runs inference; `obs` must be sized exactly input_sizes_[0] = kObsDim.
  // Returns a newly-allocated vector with the output (size = kActionDim).
  std::vector<float> infer(std::vector<float>& obs) {
    auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem, obs.data(), input_sizes_[0], input_shapes_[0].data(),
        input_shapes_[0].size()));
    auto outputs = session_->Run(
        Ort::RunOptions{nullptr}, input_names_.data(), inputs.data(),
        inputs.size(), &output_name_, 1);
    auto* data = outputs.front().GetTensorMutableData<float>();
    return std::vector<float>(data, data + output_shape_[1]);
  }

 private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::vector<std::string> input_name_strs_;
  std::vector<const char*> input_names_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<int64_t> input_sizes_;

  std::string output_name_str_;
  const char* output_name_;
  std::vector<int64_t> output_shape_;
};

// -----------------------------------------------------------------------------
// ROS2 node
// -----------------------------------------------------------------------------
class Go2VelocityNode : public rclcpp::Node {
 public:
  Go2VelocityNode() : Node("go2_velocity_node") {
    // Parameters.
    declare_parameter<std::string>("policy_path", "");
    declare_parameter<double>("standup_duration", 2.0);
    declare_parameter<double>("action_scale", 0.25);

    const std::string policy_path = get_parameter("policy_path").as_string();
    standup_duration_ = get_parameter("standup_duration").as_double();
    action_scale_ = static_cast<float>(get_parameter("action_scale").as_double());

    target_q_ = kDefaultQ;
    last_action_.fill(0.0f);

    // Load policy.
    RCLCPP_INFO(get_logger(), "Loading ONNX policy: %s", policy_path.c_str());
    policy_ = std::make_unique<OnnxPolicy>(policy_path);
    const auto& in_shape = policy_->input_shape(0);
    const auto& out_shape = policy_->output_shape();
    RCLCPP_INFO(get_logger(),
                "Policy loaded: inputs=%zu, input[0]=[%lld,%lld], output=[%lld,%lld]",
                policy_->input_count(),
                (long long)in_shape[0], (long long)in_shape[1],
                (long long)out_shape[0], (long long)out_shape[1]);
    if (in_shape.size() != 2 || in_shape[1] != kObsDim ||
        out_shape.size() != 2 || out_shape[1] != kActionDim) {
      RCLCPP_FATAL(get_logger(),
                   "Policy shape mismatch: expected input [*,%d] and output [*,%d]",
                   kObsDim, kActionDim);
      throw std::runtime_error("policy shape mismatch");
    }
    obs_buffer_.assign(kObsDim, 0.0f);

    // QoS: best-effort, depth 1.
    auto qos = rclcpp::SensorDataQoS().keep_last(1);

    // Subscribers.
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/imu", qos,
        [this](sensor_msgs::msg::Imu::SharedPtr msg) { ImuCb(*msg); });
    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", qos,
        [this](sensor_msgs::msg::JointState::SharedPtr msg) { JointStateCb(*msg); });
    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", qos,
        [this](geometry_msgs::msg::Twist::SharedPtr msg) { CmdVelCb(*msg); });

    // Publishers.
    joint_cmd_pub_ = create_publisher<sensor_msgs::msg::JointState>("/joint_commands", qos);
    policy_active_pub_ = create_publisher<std_msgs::msg::Bool>("/policy_active", qos);

    // 500 Hz publish timer (2 ms period).
    publish_timer_ = create_wall_timer(
        std::chrono::microseconds(2000),
        [this]() { PublishTimerCb(); });
    // 50 Hz policy timer (20 ms period).
    policy_timer_ = create_wall_timer(
        std::chrono::microseconds(20000),
        [this]() { PolicyTimerCb(); });

    RCLCPP_INFO(get_logger(),
                "go2_velocity_node started (standup=%.2fs)",
                standup_duration_);
    RCLCPP_INFO(get_logger(), "Waiting for /joint_states...");
  }

 private:
  enum class State { kWaitingForState, kStandup, kPolicy };

  // --- Callbacks ---

  void ImuCb(const sensor_msgs::msg::Imu& msg) {
    std::lock_guard<std::mutex> lock(state_mu_);
    // sensor_msgs quaternion: x,y,z,w -> internal: w,x,y,z
    latest_quat_[0] = msg.orientation.w;
    latest_quat_[1] = msg.orientation.x;
    latest_quat_[2] = msg.orientation.y;
    latest_quat_[3] = msg.orientation.z;
    latest_gyro_[0] = msg.angular_velocity.x;
    latest_gyro_[1] = msg.angular_velocity.y;
    latest_gyro_[2] = msg.angular_velocity.z;
  }

  void JointStateCb(const sensor_msgs::msg::JointState& msg) {
    std::lock_guard<std::mutex> lock(state_mu_);
    for (size_t j = 0; j < msg.name.size() && j < msg.position.size(); ++j) {
      auto it = kNameToIndex.find(msg.name[j]);
      if (it == kNameToIndex.end()) continue;
      int idx = it->second;
      latest_q_[idx] = msg.position[j];
      if (j < msg.velocity.size()) latest_dq_[idx] = msg.velocity[j];
    }
    if (state_ == State::kWaitingForState) {
      q_init_ = latest_q_;
      standup_start_time_ = std::chrono::steady_clock::now();
      state_ = State::kStandup;
      RCLCPP_INFO(get_logger(), "Received first /joint_states, starting standup.");
    }
  }

  void CmdVelCb(const geometry_msgs::msg::Twist& msg) {
    std::lock_guard<std::mutex> lock(cmd_mu_);
    cmd_lin_x_ = msg.linear.x;
    cmd_lin_y_ = msg.linear.y;
    cmd_ang_z_ = msg.angular.z;
  }

  // --- Timers ---

  void PublishTimerCb() {
    State state_snapshot;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      state_snapshot = state_;
    }
    if (state_snapshot == State::kWaitingForState) return;

    // Build JointState message with current targets and joint names.
    sensor_msgs::msg::JointState cmd;
    cmd.header.stamp = now();
    cmd.name = kJointNames;
    cmd.position.resize(kNumLegMotor);

    if (state_snapshot == State::kStandup) {
      double t = std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - standup_start_time_)
                     .count();
      double alpha = std::clamp(t / standup_duration_, 0.0, 1.0);
      for (int i = 0; i < kNumLegMotor; ++i) {
        cmd.position[i] =
            (1.0f - alpha) * q_init_[i] + alpha * kDefaultQ[i];
      }
      if (alpha >= 1.0) {
        std::lock_guard<std::mutex> lock(state_mu_);
        if (state_ == State::kStandup) {
          state_ = State::kPolicy;
          last_action_.fill(0.0f);
          target_q_ = kDefaultQ;
          policy_start_time_ = std::chrono::steady_clock::now();
          RCLCPP_INFO(get_logger(), "Standup complete, entering POLICY mode.");
        }
      }
    } else {  // kPolicy
      std::lock_guard<std::mutex> lock(target_mu_);
      for (int i = 0; i < kNumLegMotor; ++i) {
        cmd.position[i] = target_q_[i];
      }
    }

    joint_cmd_pub_->publish(cmd);

    // Signal which gains the bridge should use.
    std_msgs::msg::Bool active_msg;
    active_msg.data = (state_snapshot == State::kPolicy);
    policy_active_pub_->publish(active_msg);
  }

  void PolicyTimerCb() {
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      if (state_ != State::kPolicy) return;
    }

    BuildObservation(obs_buffer_);

    std::vector<float> action;
    try {
      action = policy_->infer(obs_buffer_);
    } catch (const Ort::Exception& e) {
      RCLCPP_ERROR(get_logger(), "ONNX inference failed: %s", e.what());
      return;
    }

    // Cache raw action for the next observation.
    for (int i = 0; i < kActionDim; ++i) last_action_[i] = action[i];

    // target[i] = raw[i] * scale + default[i]
    std::array<float, kNumLegMotor> target;
    for (int i = 0; i < kNumLegMotor; ++i) {
      target[i] = action[i] * action_scale_ + kDefaultQ[i];
    }
    {
      std::lock_guard<std::mutex> lock(target_mu_);
      target_q_ = target;
    }
  }

  // --- Observation builder ---

  void BuildObservation(std::vector<float>& obs) {
    std::array<float, kNumLegMotor> q, dq;
    std::array<float, 4> quat;
    std::array<float, 3> gyro;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      q = latest_q_;
      dq = latest_dq_;
      quat = latest_quat_;
      gyro = latest_gyro_;
    }

    double cx, cy, cz;
    {
      std::lock_guard<std::mutex> lock(cmd_mu_);
      cx = cmd_lin_x_;
      cy = cmd_lin_y_;
      cz = cmd_ang_z_;
    }

    // 0..2: base_ang_vel (gyro, unscaled)
    obs[0] = gyro[0];
    obs[1] = gyro[1];
    obs[2] = gyro[2];

    // 3..5: projected gravity
    float g_body[3];
    projected_gravity(quat[0], quat[1], quat[2], quat[3], g_body);
    obs[3] = g_body[0];
    obs[4] = g_body[1];
    obs[5] = g_body[2];

    // 6..8: command
    obs[6] = static_cast<float>(cx);
    obs[7] = static_cast<float>(cy);
    obs[8] = static_cast<float>(cz);

    // 9..10: phase (sin, cos); zero when standing.
    double t = std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - policy_start_time_)
                   .count();
    double cmd_norm = std::sqrt(cx * cx + cy * cy + cz * cz);
    if (cmd_norm < kCmdStandThreshold) {
      obs[9] = 0.0f;
      obs[10] = 0.0f;
    } else {
      double gp = std::fmod(t, kPhasePeriod) / kPhasePeriod;
      obs[9] = static_cast<float>(std::sin(gp * 2.0 * M_PI));
      obs[10] = static_cast<float>(std::cos(gp * 2.0 * M_PI));
    }

    // 11..22: joint_pos_rel
    // 23..34: joint_vel
    for (int i = 0; i < kNumLegMotor; ++i) {
      obs[11 + i] = q[i] - kDefaultQ[i];
      obs[23 + i] = dq[i];
    }

    // 35..46: last_action (raw)
    for (int i = 0; i < kActionDim; ++i) {
      obs[35 + i] = last_action_[i];
    }
  }

  // --- State ---
  std::unique_ptr<OnnxPolicy> policy_;
  std::vector<float> obs_buffer_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr policy_active_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  rclcpp::TimerBase::SharedPtr policy_timer_;

  double standup_duration_;
  float action_scale_;

  std::array<float, kNumLegMotor> target_q_{};
  std::array<float, kNumLegMotor> q_init_{};
  std::array<float, kActionDim> last_action_{};

  // Latest sensor data (protected by state_mu_).
  std::array<float, kNumLegMotor> latest_q_{};
  std::array<float, kNumLegMotor> latest_dq_{};
  std::array<float, 4> latest_quat_{1.0f, 0.0f, 0.0f, 0.0f};
  std::array<float, 3> latest_gyro_{};

  // Latest cmd_vel (protected by cmd_mu_).
  double cmd_lin_x_ = 0.0;
  double cmd_lin_y_ = 0.0;
  double cmd_ang_z_ = 0.0;

  State state_ = State::kWaitingForState;
  std::chrono::steady_clock::time_point standup_start_time_;
  std::chrono::steady_clock::time_point policy_start_time_;

  std::mutex state_mu_;
  std::mutex cmd_mu_;
  std::mutex target_mu_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Go2VelocityNode>());
  rclcpp::shutdown();
  return 0;
}
