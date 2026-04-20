// Go2 velocity policy deployment node.
//
// Subscribes: /lowstate (unitree_go::msg::LowState), /cmd_vel (geometry_msgs::msg::Twist)
// Publishes:  /lowcmd   (unitree_go::msg::LowCmd) at 500 Hz
//
// Runs an ONNX policy trained by unitree_rl_mjlab on the Unitree-Go2-Flat task.
// The policy runs at 50 Hz; the publisher runs at 500 Hz with zero-order hold
// between inferences.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "unitree_go/msg/low_state.hpp"
#include "unitree_go/msg/low_cmd.hpp"

#include "motor_crc.hpp"

namespace {

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
constexpr int kNumLegMotor = 12;
constexpr int kNumMotor = 20;            // LowCmd has 20 motor slots
constexpr int kObsDim = 47;              // 3 + 3 + 3 + 2 + 12 + 12 + 12
constexpr int kActionDim = 12;
constexpr double kPhasePeriod = 0.6;     // seconds, matches training
constexpr double kCmdStandThreshold = 0.1;

// Joint remap:   isaac[i] = sdk[kJointMap[i]]
//                sdk[kJointMap[i]] = isaac[i]
// Isaac Lab order: FL, FR, RL, RR  (policy side)
// Unitree SDK order: FR, FL, RR, RL  (motor_state indices)
// Self-inverse permutation (swaps left<->right on each axle).
constexpr std::array<int, kNumLegMotor> kJointMap = {
    3, 4, 5,   // FL_hip, FL_thigh, FL_calf   -> sdk 3,4,5
    0, 1, 2,   // FR_hip, FR_thigh, FR_calf   -> sdk 0,1,2
    9, 10, 11, // RL_hip, RL_thigh, RL_calf   -> sdk 9,10,11
    6, 7, 8    // RR_hip, RR_thigh, RR_calf   -> sdk 6,7,8
};

// Default joint positions in Isaac Lab order (from go2_constants.py and from
// the default_joint_pos metadata embedded in the exported ONNX).
constexpr std::array<float, kNumLegMotor> kDefaultQIsaac = {
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
    // Parameters
    declare_parameter<std::string>("policy_path", "");
    declare_parameter<bool>("real_robot", false);
    declare_parameter<double>("standup_duration", 3.0);
    declare_parameter<double>("action_scale", 0.25);
    declare_parameter<double>("standup_kp", 60.0);
    declare_parameter<double>("standup_kd", 5.0);
    declare_parameter<double>("kp_hip", 20.0);
    declare_parameter<double>("kp_thigh", 20.0);
    declare_parameter<double>("kp_calf", 40.0);
    declare_parameter<double>("kd_hip", 1.0);
    declare_parameter<double>("kd_thigh", 1.0);
    declare_parameter<double>("kd_calf", 2.0);

    const std::string policy_path = get_parameter("policy_path").as_string();
    real_robot_ = get_parameter("real_robot").as_bool();
    standup_duration_ = get_parameter("standup_duration").as_double();
    action_scale_ = static_cast<float>(get_parameter("action_scale").as_double());
    standup_kp_ = static_cast<float>(get_parameter("standup_kp").as_double());
    standup_kd_ = static_cast<float>(get_parameter("standup_kd").as_double());

    // Per-joint policy gains in SDK order. SDK and Isaac share the same
    // (hip, thigh, calf) pattern per leg — only the leg order differs — so the
    // same per-index pattern applies to both orderings.
    {
      float kp_hip = get_parameter("kp_hip").as_double();
      float kp_thigh = get_parameter("kp_thigh").as_double();
      float kp_calf = get_parameter("kp_calf").as_double();
      float kd_hip = get_parameter("kd_hip").as_double();
      float kd_thigh = get_parameter("kd_thigh").as_double();
      float kd_calf = get_parameter("kd_calf").as_double();
      for (int leg = 0; leg < 4; ++leg) {
        policy_kp_sdk_[leg * 3 + 0] = kp_hip;
        policy_kp_sdk_[leg * 3 + 1] = kp_thigh;
        policy_kp_sdk_[leg * 3 + 2] = kp_calf;
        policy_kd_sdk_[leg * 3 + 0] = kd_hip;
        policy_kd_sdk_[leg * 3 + 1] = kd_thigh;
        policy_kd_sdk_[leg * 3 + 2] = kd_calf;
      }
    }

    // Default joint positions in SDK order.
    for (int i = 0; i < kNumLegMotor; ++i) {
      q_default_sdk_[kJointMap[i]] = kDefaultQIsaac[i];
    }
    target_q_sdk_ = q_default_sdk_;
    last_action_.fill(0.0f);

    // Load policy
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

    auto qos = rclcpp::SensorDataQoS().keep_last(1);

    lowstate_sub_ = create_subscription<unitree_go::msg::LowState>(
        "/lowstate", qos,
        [this](unitree_go::msg::LowState::SharedPtr msg) { LowStateCb(*msg); });
    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", qos,
        [this](geometry_msgs::msg::Twist::SharedPtr msg) { CmdVelCb(*msg); });

    lowcmd_pub_ = create_publisher<unitree_go::msg::LowCmd>("/lowcmd", qos);

    publish_timer_ = create_wall_timer(
        std::chrono::microseconds(2000),
        [this]() { PublishTimerCb(); });
    policy_timer_ = create_wall_timer(
        std::chrono::microseconds(20000),
        [this]() { PolicyTimerCb(); });

    RCLCPP_INFO(get_logger(),
                "go2_velocity_node started (real_robot=%s, standup=%.2fs)",
                real_robot_ ? "true" : "false", standup_duration_);
    RCLCPP_INFO(get_logger(), "Waiting for /lowstate...");
  }

 private:
  enum class State { kWaitingForState, kStandup, kPolicy };

  // --- Callbacks ---

  void LowStateCb(const unitree_go::msg::LowState& msg) {
    std::lock_guard<std::mutex> lock(state_mu_);
    for (int i = 0; i < kNumLegMotor; ++i) {
      latest_q_sdk_[i] = msg.motor_state[i].q;
      latest_dq_sdk_[i] = msg.motor_state[i].dq;
    }
    // Unitree stores quaternion as (w, x, y, z).
    latest_quat_[0] = msg.imu_state.quaternion[0];
    latest_quat_[1] = msg.imu_state.quaternion[1];
    latest_quat_[2] = msg.imu_state.quaternion[2];
    latest_quat_[3] = msg.imu_state.quaternion[3];
    latest_gyro_[0] = msg.imu_state.gyroscope[0];
    latest_gyro_[1] = msg.imu_state.gyroscope[1];
    latest_gyro_[2] = msg.imu_state.gyroscope[2];

    if (state_ == State::kWaitingForState) {
      q_init_sdk_ = latest_q_sdk_;
      standup_start_time_ = std::chrono::steady_clock::now();
      state_ = State::kStandup;
      RCLCPP_INFO(get_logger(), "Received first /lowstate, starting standup.");
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

    unitree_go::msg::LowCmd cmd{};
    cmd.head[0] = 0xFE;
    cmd.head[1] = 0xEF;
    cmd.level_flag = 0xFF;
    cmd.gpio = 0;

    // Safe-stop defaults for all 20 motor slots.
    for (int i = 0; i < kNumMotor; ++i) {
      cmd.motor_cmd[i].mode = 0x01;
      cmd.motor_cmd[i].q = static_cast<float>(PosStopF);
      cmd.motor_cmd[i].dq = static_cast<float>(VelStopF);
      cmd.motor_cmd[i].tau = 0.0f;
      cmd.motor_cmd[i].kp = 0.0f;
      cmd.motor_cmd[i].kd = 0.0f;
    }

    if (state_snapshot == State::kStandup) {
      double t = std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - standup_start_time_)
                     .count();
      double alpha = std::clamp(t / standup_duration_, 0.0, 1.0);
      for (int i = 0; i < kNumLegMotor; ++i) {
        float q_target =
            (1.0f - alpha) * q_init_sdk_[i] + alpha * q_default_sdk_[i];
        cmd.motor_cmd[i].mode = 0x01;
        cmd.motor_cmd[i].q = q_target;
        cmd.motor_cmd[i].dq = 0.0f;
        cmd.motor_cmd[i].tau = 0.0f;
        cmd.motor_cmd[i].kp = standup_kp_;
        cmd.motor_cmd[i].kd = standup_kd_;
      }
      if (alpha >= 1.0) {
        std::lock_guard<std::mutex> lock(state_mu_);
        if (state_ == State::kStandup) {
          state_ = State::kPolicy;
          last_action_.fill(0.0f);
          target_q_sdk_ = q_default_sdk_;
          policy_start_time_ = std::chrono::steady_clock::now();
          RCLCPP_INFO(get_logger(), "Standup complete, entering POLICY mode.");
        }
      }
    } else {  // kPolicy
      std::array<float, kNumLegMotor> target;
      {
        std::lock_guard<std::mutex> lock(target_mu_);
        target = target_q_sdk_;
      }
      for (int i = 0; i < kNumLegMotor; ++i) {
        cmd.motor_cmd[i].mode = 0x01;
        cmd.motor_cmd[i].q = target[i];
        cmd.motor_cmd[i].dq = 0.0f;
        cmd.motor_cmd[i].tau = 0.0f;
        cmd.motor_cmd[i].kp = policy_kp_sdk_[i];
        cmd.motor_cmd[i].kd = policy_kd_sdk_[i];
      }
    }

    if (real_robot_) {
      get_crc(cmd);
    }

    lowcmd_pub_->publish(cmd);
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

    for (int i = 0; i < kActionDim; ++i) last_action_[i] = action[i];

    std::array<float, kNumLegMotor> target_sdk;
    for (int i = 0; i < kNumLegMotor; ++i) {
      float q_isaac = action[i] * action_scale_ + kDefaultQIsaac[i];
      target_sdk[kJointMap[i]] = q_isaac;
    }
    {
      std::lock_guard<std::mutex> lock(target_mu_);
      target_q_sdk_ = target_sdk;
    }
  }

  // --- Observation builder ---

  void BuildObservation(std::vector<float>& obs) {
    std::array<float, kNumLegMotor> q_sdk, dq_sdk;
    std::array<float, 4> quat;
    std::array<float, 3> gyro;
    {
      std::lock_guard<std::mutex> lock(state_mu_);
      q_sdk = latest_q_sdk_;
      dq_sdk = latest_dq_sdk_;
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

    // 0..2: base_ang_vel
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

    // 11..22: joint_pos_rel (Isaac order)
    // 23..34: joint_vel     (Isaac order)
    for (int i = 0; i < kNumLegMotor; ++i) {
      int sdk_idx = kJointMap[i];
      obs[11 + i] = q_sdk[sdk_idx] - kDefaultQIsaac[i];
      obs[23 + i] = dq_sdk[sdk_idx];
    }

    // 35..46: last_action (Isaac order)
    for (int i = 0; i < kActionDim; ++i) {
      obs[35 + i] = last_action_[i];
    }
  }

  // --- State ---
  std::unique_ptr<OnnxPolicy> policy_;
  std::vector<float> obs_buffer_;

  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr lowstate_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr lowcmd_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  rclcpp::TimerBase::SharedPtr policy_timer_;

  bool real_robot_;
  double standup_duration_;
  float action_scale_;
  float standup_kp_, standup_kd_;
  std::array<float, kNumLegMotor> policy_kp_sdk_;
  std::array<float, kNumLegMotor> policy_kd_sdk_;

  std::array<float, kNumLegMotor> q_default_sdk_;
  std::array<float, kNumLegMotor> q_init_sdk_{};
  std::array<float, kNumLegMotor> target_q_sdk_{};
  std::array<float, kActionDim> last_action_{};

  // Latest LowState (protected by state_mu_).
  std::array<float, kNumLegMotor> latest_q_sdk_{};
  std::array<float, kNumLegMotor> latest_dq_sdk_{};
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
