// Go2 hardware bridge node.
//
// Translates between standard ROS2 messages and Unitree-specific messages:
//   /lowstate  (unitree_go)  -->  /imu  (sensor_msgs)  +  /joint_states (sensor_msgs)
//   /joint_commands (sensor_msgs)  +  /policy_active (std_msgs)  -->  /lowcmd (unitree_go)
//
// Gains are read from config (gains.yaml).  The /policy_active topic selects
// between standup gains (high stiffness) and per-joint policy gains.

#include <array>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/bool.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/low_state.hpp"

#include "motor_crc.hpp"

namespace {

constexpr int kNumLegMotor = 12;
constexpr int kNumMotor = 20;

// SDK motor index -> joint name (for publishing /joint_states).
const std::vector<std::string> kSdkJointNames = {
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
};

// Joint name -> SDK motor index (for receiving /joint_commands).
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
    // Parameters.
    declare_parameter<bool>("real_robot", false);
    declare_parameter<double>("standup_kp", 60.0);
    declare_parameter<double>("standup_kd", 5.0);
    declare_parameter<double>("kp_hip", 20.0);
    declare_parameter<double>("kp_thigh", 20.0);
    declare_parameter<double>("kp_calf", 40.0);
    declare_parameter<double>("kd_hip", 1.0);
    declare_parameter<double>("kd_thigh", 1.0);
    declare_parameter<double>("kd_calf", 2.0);

    real_robot_ = get_parameter("real_robot").as_bool();
    standup_kp_ = static_cast<float>(get_parameter("standup_kp").as_double());
    standup_kd_ = static_cast<float>(get_parameter("standup_kd").as_double());

    // Build per-SDK-index policy gains from joint names.
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

    // QoS: best-effort, depth 1.
    auto qos = rclcpp::SensorDataQoS().keep_last(1);

    // Subscribers.
    lowstate_sub_ = create_subscription<unitree_go::msg::LowState>(
        "/lowstate", qos,
        [this](unitree_go::msg::LowState::SharedPtr msg) { LowStateCb(*msg); });
    joint_cmd_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        "/joint_commands", qos,
        [this](sensor_msgs::msg::JointState::SharedPtr msg) { JointCmdCb(*msg); });
    policy_active_sub_ = create_subscription<std_msgs::msg::Bool>(
        "/policy_active", qos,
        [this](std_msgs::msg::Bool::SharedPtr msg) { PolicyActiveCb(*msg); });

    // Publishers.
    imu_pub_ = create_publisher<sensor_msgs::msg::Imu>("/imu", qos);
    joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>("/joint_states", qos);
    lowcmd_pub_ = create_publisher<unitree_go::msg::LowCmd>("/lowcmd", qos);

    // 500 Hz timer for /lowcmd (2 ms period).
    publish_timer_ = create_wall_timer(
        std::chrono::microseconds(2000),
        [this]() { PublishTimerCb(); });

    RCLCPP_INFO(get_logger(),
                "go2_bridge_node started (real_robot=%s)",
                real_robot_ ? "true" : "false");
  }

 private:
  // --- Callbacks ---

  void LowStateCb(const unitree_go::msg::LowState& msg) {
    // Publish /joint_states — straight pass-through with names.
    sensor_msgs::msg::JointState js;
    js.header.stamp = now();
    js.name = kSdkJointNames;
    js.position.resize(kNumLegMotor);
    js.velocity.resize(kNumLegMotor);
    for (int i = 0; i < kNumLegMotor; ++i) {
      js.position[i] = msg.motor_state[i].q;
      js.velocity[i] = msg.motor_state[i].dq;
    }
    joint_state_pub_->publish(js);

    // Publish /imu.
    sensor_msgs::msg::Imu imu;
    imu.header.stamp = js.header.stamp;
    imu.header.frame_id = "imu_link";
    // Unitree quaternion: w,x,y,z -> sensor_msgs: x,y,z,w fields.
    imu.orientation.w = msg.imu_state.quaternion[0];
    imu.orientation.x = msg.imu_state.quaternion[1];
    imu.orientation.y = msg.imu_state.quaternion[2];
    imu.orientation.z = msg.imu_state.quaternion[3];
    imu.angular_velocity.x = msg.imu_state.gyroscope[0];
    imu.angular_velocity.y = msg.imu_state.gyroscope[1];
    imu.angular_velocity.z = msg.imu_state.gyroscope[2];
    imu.linear_acceleration.x = msg.imu_state.accelerometer[0];
    imu.linear_acceleration.y = msg.imu_state.accelerometer[1];
    imu.linear_acceleration.z = msg.imu_state.accelerometer[2];
    imu_pub_->publish(imu);
  }

  void JointCmdCb(const sensor_msgs::msg::JointState& msg) {
    std::lock_guard<std::mutex> lock(cmd_mu_);
    for (size_t j = 0; j < msg.name.size() && j < msg.position.size(); ++j) {
      auto it = kNameToSdk.find(msg.name[j]);
      if (it == kNameToSdk.end()) continue;
      latest_target_sdk_[it->second] = static_cast<float>(msg.position[j]);
    }
    has_command_ = true;
  }

  void PolicyActiveCb(const std_msgs::msg::Bool& msg) {
    std::lock_guard<std::mutex> lock(cmd_mu_);
    policy_active_ = msg.data;
  }

  // --- 500 Hz timer ---

  void PublishTimerCb() {
    std::array<float, kNumLegMotor> target_sdk;
    bool has_cmd;
    bool use_policy_gains;
    {
      std::lock_guard<std::mutex> lock(cmd_mu_);
      has_cmd = has_command_;
      target_sdk = latest_target_sdk_;
      use_policy_gains = policy_active_;
    }

    if (!has_cmd) return;

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

    // Fill leg motor commands.
    for (int i = 0; i < kNumLegMotor; ++i) {
      cmd.motor_cmd[i].mode = 0x01;
      cmd.motor_cmd[i].q = target_sdk[i];
      cmd.motor_cmd[i].dq = 0.0f;
      cmd.motor_cmd[i].tau = 0.0f;
      if (use_policy_gains) {
        cmd.motor_cmd[i].kp = policy_kp_sdk_[i];
        cmd.motor_cmd[i].kd = policy_kd_sdk_[i];
      } else {
        cmd.motor_cmd[i].kp = standup_kp_;
        cmd.motor_cmd[i].kd = standup_kd_;
      }
    }

    if (real_robot_) {
      get_crc(cmd);
    }

    lowcmd_pub_->publish(cmd);
  }

  // --- State ---

  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr lowstate_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr policy_active_sub_;

  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr lowcmd_pub_;

  rclcpp::TimerBase::SharedPtr publish_timer_;

  bool real_robot_;
  float standup_kp_, standup_kd_;
  std::array<float, kNumLegMotor> policy_kp_sdk_;
  std::array<float, kNumLegMotor> policy_kd_sdk_;

  // Protected by cmd_mu_.
  std::mutex cmd_mu_;
  std::array<float, kNumLegMotor> latest_target_sdk_{};
  bool has_command_ = false;
  bool policy_active_ = false;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Go2BridgeNode>());
  rclcpp::shutdown();
  return 0;
}
