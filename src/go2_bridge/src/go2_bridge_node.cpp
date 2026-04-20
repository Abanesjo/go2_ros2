// Go2 sim-side bridge node.
//
// Translates between standard ROS2 messages (published/consumed by a simulator)
// and the Unitree-specific messages (published/consumed by go2_velocity_node):
//
//   /joint_states (sensor_msgs) + /imu (sensor_msgs) --> /lowstate (unitree_go)
//   /lowcmd       (unitree_go)                        --> /joint_commands
//                                                         (sensor_msgs, effort = torque)
//
// The bridge runs at 500 Hz and acts as a simulated motor controller: it applies
// a PD law using gains from gains.yaml to turn /lowcmd target positions into
// joint torques for the simulator.
//
// Standup vs. policy mode is inferred from /lowcmd.motor_cmd[0].kp: if it
// matches standup_kp (within tolerance), the bridge uses the uniform standup
// gains; otherwise it uses the per-joint-type policy gains.

#include <array>
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/low_state.hpp"

namespace {

constexpr int kNumLegMotor = 12;

// SDK motor index -> joint name.
const std::vector<std::string> kSdkJointNames = {
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
};

// Joint name -> SDK motor index.
const std::unordered_map<std::string, int> kNameToSdk = {
    {"FR_hip_joint", 0}, {"FR_thigh_joint", 1}, {"FR_calf_joint", 2},
    {"FL_hip_joint", 3}, {"FL_thigh_joint", 4}, {"FL_calf_joint", 5},
    {"RR_hip_joint", 6}, {"RR_thigh_joint", 7}, {"RR_calf_joint", 8},
    {"RL_hip_joint", 9}, {"RL_thigh_joint", 10}, {"RL_calf_joint", 11},
};

}  // namespace

class Go2BridgeNode : public rclcpp::Node {
 public:
  Go2BridgeNode() : Node("go2_bridge_node") {
    declare_parameter<double>("standup_kp", 60.0);
    declare_parameter<double>("standup_kd", 5.0);
    declare_parameter<double>("kp_hip", 20.0);
    declare_parameter<double>("kp_thigh", 20.0);
    declare_parameter<double>("kp_calf", 40.0);
    declare_parameter<double>("kd_hip", 1.0);
    declare_parameter<double>("kd_thigh", 1.0);
    declare_parameter<double>("kd_calf", 2.0);

    standup_kp_ = static_cast<float>(get_parameter("standup_kp").as_double());
    standup_kd_ = static_cast<float>(get_parameter("standup_kd").as_double());

    float kp_hip = get_parameter("kp_hip").as_double();
    float kp_thigh = get_parameter("kp_thigh").as_double();
    float kp_calf = get_parameter("kp_calf").as_double();
    float kd_hip = get_parameter("kd_hip").as_double();
    float kd_thigh = get_parameter("kd_thigh").as_double();
    float kd_calf = get_parameter("kd_calf").as_double();
    for (int i = 0; i < kNumLegMotor; ++i) {
      const auto& name = kSdkJointNames[i];
      if (name.find("hip") != std::string::npos) {
        policy_kp_sdk_[i] = kp_hip;
        policy_kd_sdk_[i] = kd_hip;
      } else if (name.find("thigh") != std::string::npos) {
        policy_kp_sdk_[i] = kp_thigh;
        policy_kd_sdk_[i] = kd_thigh;
      } else {
        policy_kp_sdk_[i] = kp_calf;
        policy_kd_sdk_[i] = kd_calf;
      }
    }

    auto qos = rclcpp::SensorDataQoS().keep_last(1);

    joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", qos,
        [this](sensor_msgs::msg::JointState::SharedPtr msg) { JointStatesCb(*msg); });
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/imu", qos,
        [this](sensor_msgs::msg::Imu::SharedPtr msg) { ImuCb(*msg); });
    lowcmd_sub_ = create_subscription<unitree_go::msg::LowCmd>(
        "/lowcmd", qos,
        [this](unitree_go::msg::LowCmd::SharedPtr msg) { LowCmdCb(*msg); });

    lowstate_pub_ = create_publisher<unitree_go::msg::LowState>("/lowstate", qos);
    joint_cmd_pub_ = create_publisher<sensor_msgs::msg::JointState>("/joint_commands", qos);

    timer_ = create_wall_timer(
        std::chrono::microseconds(2000),
        [this]() { TimerCb(); });

    RCLCPP_INFO(get_logger(), "go2_bridge_node started (sim-side, 500 Hz torque PD)");
  }

 private:
  // --- Callbacks ---

  void JointStatesCb(const sensor_msgs::msg::JointState& msg) {
    std::lock_guard<std::mutex> lock(mu_);
    for (size_t j = 0; j < msg.name.size() && j < msg.position.size(); ++j) {
      auto it = kNameToSdk.find(msg.name[j]);
      if (it == kNameToSdk.end()) continue;
      int s = it->second;
      q_sdk_[s] = static_cast<float>(msg.position[j]);
      if (j < msg.velocity.size()) dq_sdk_[s] = static_cast<float>(msg.velocity[j]);
    }
    has_joint_states_ = true;
  }

  void ImuCb(const sensor_msgs::msg::Imu& msg) {
    std::lock_guard<std::mutex> lock(mu_);
    // Unitree stores quaternion as (w, x, y, z).
    quat_[0] = msg.orientation.w;
    quat_[1] = msg.orientation.x;
    quat_[2] = msg.orientation.y;
    quat_[3] = msg.orientation.z;
    gyro_[0] = msg.angular_velocity.x;
    gyro_[1] = msg.angular_velocity.y;
    gyro_[2] = msg.angular_velocity.z;
    accel_[0] = msg.linear_acceleration.x;
    accel_[1] = msg.linear_acceleration.y;
    accel_[2] = msg.linear_acceleration.z;
    has_imu_ = true;
  }

  void LowCmdCb(const unitree_go::msg::LowCmd& msg) {
    std::lock_guard<std::mutex> lock(mu_);
    for (int s = 0; s < kNumLegMotor; ++s) {
      target_q_sdk_[s] = msg.motor_cmd[s].q;
    }
    latest_lowcmd_kp_ = msg.motor_cmd[0].kp;
    has_lowcmd_ = true;
  }

  // --- 500 Hz timer ---

  void TimerCb() {
    std::array<float, kNumLegMotor> q, dq, target_q;
    std::array<float, 4> quat;
    std::array<float, 3> gyro, accel;
    bool have_state, have_imu, have_cmd;
    float lowcmd_kp;
    {
      std::lock_guard<std::mutex> lock(mu_);
      have_state = has_joint_states_;
      have_imu = has_imu_;
      have_cmd = has_lowcmd_;
      q = q_sdk_;
      dq = dq_sdk_;
      target_q = target_q_sdk_;
      quat = quat_;
      gyro = gyro_;
      accel = accel_;
      lowcmd_kp = latest_lowcmd_kp_;
    }

    if (!have_state || !have_imu) return;

    // --- Publish synthesized /lowstate ---
    unitree_go::msg::LowState ls{};
    for (int s = 0; s < kNumLegMotor; ++s) {
      ls.motor_state[s].q = q[s];
      ls.motor_state[s].dq = dq[s];
    }
    for (int i = 0; i < 4; ++i) ls.imu_state.quaternion[i] = quat[i];
    for (int i = 0; i < 3; ++i) {
      ls.imu_state.gyroscope[i] = gyro[i];
      ls.imu_state.accelerometer[i] = accel[i];
    }
    lowstate_pub_->publish(ls);

    // --- Publish /joint_commands (torques) ---
    if (!have_cmd) return;

    const bool is_standup = std::fabs(lowcmd_kp - standup_kp_) < 0.5f;

    sensor_msgs::msg::JointState jc;
    jc.header.stamp = now();
    jc.name = kSdkJointNames;
    jc.effort.resize(kNumLegMotor);
    for (int s = 0; s < kNumLegMotor; ++s) {
      const float kp = is_standup ? standup_kp_ : policy_kp_sdk_[s];
      const float kd = is_standup ? standup_kd_ : policy_kd_sdk_[s];
      jc.effort[s] = kp * (target_q[s] - q[s]) - kd * dq[s];
    }
    joint_cmd_pub_->publish(jc);
  }

  // --- Subscriptions / publications / timer ---
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<unitree_go::msg::LowCmd>::SharedPtr lowcmd_sub_;

  rclcpp::Publisher<unitree_go::msg::LowState>::SharedPtr lowstate_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_pub_;

  rclcpp::TimerBase::SharedPtr timer_;

  // --- Gain parameters ---
  float standup_kp_, standup_kd_;
  std::array<float, kNumLegMotor> policy_kp_sdk_{};
  std::array<float, kNumLegMotor> policy_kd_sdk_{};

  // --- Cached state (protected by mu_) ---
  std::mutex mu_;
  std::array<float, kNumLegMotor> q_sdk_{};
  std::array<float, kNumLegMotor> dq_sdk_{};
  std::array<float, kNumLegMotor> target_q_sdk_{};
  std::array<float, 4> quat_{1.0f, 0.0f, 0.0f, 0.0f};
  std::array<float, 3> gyro_{};
  std::array<float, 3> accel_{};
  float latest_lowcmd_kp_ = 0.0f;
  bool has_joint_states_ = false;
  bool has_imu_ = false;
  bool has_lowcmd_ = false;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Go2BridgeNode>());
  rclcpp::shutdown();
  return 0;
}
