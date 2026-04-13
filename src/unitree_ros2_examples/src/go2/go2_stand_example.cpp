#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <pthread.h>
#include <sched.h>

#include "motor_crc.h"
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/low_state.hpp"

class LowLevelCmdNode : public rclcpp::Node {
 public:
  explicit LowLevelCmdNode() : Node("low_level_cmd_node") {
    InitLowCmd();

    auto qos = rclcpp::QoS(1)
                   .reliability(rclcpp::ReliabilityPolicy::BestEffort)
                   .durability(rclcpp::DurabilityPolicy::Volatile);

    low_cmd_pub_ =
        this->create_publisher<unitree_go::msg::LowCmd>("/lowcmd", qos);

    state_cbg_ = this->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = state_cbg_;

    low_state_sub_ = this->create_subscription<unitree_go::msg::LowState>(
        "/lowstate", qos,
        [this](const unitree_go::msg::LowState::SharedPtr msg) {
          low_state_ = *msg;
          state_received_ = true;
        },
        sub_opts);
  }

  void StartCmdLoop() {
    cmd_thread_ = std::thread([this]() {
      // Attempt RT scheduling — non-fatal if it fails
      struct sched_param param;
      param.sched_priority = 50;
      if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
        RCLCPP_WARN(get_logger(),
                    "Could not set RT priority (run with 'sudo chrt -f 50' or "
                    "'--cap-add=SYS_NICE' in Docker if needed)");
      }

      const auto period = std::chrono::microseconds(2000);
      auto next = std::chrono::steady_clock::now();

      while (rclcpp::ok()) {
        LowCmdWrite();
        next += period;
        std::this_thread::sleep_until(next);
      }
    });
  }

  ~LowLevelCmdNode() {
    if (cmd_thread_.joinable()) cmd_thread_.join();
  }

 private:
  void InitLowCmd() {
    low_cmd_.head[0] = 0xFE;
    low_cmd_.head[1] = 0xEF;
    low_cmd_.level_flag = 0xFF;
    low_cmd_.gpio = 0;

    for (int i = 0; i < 20; i++) {
      low_cmd_.motor_cmd[i].mode = 0x01;  // servo (PMSM) mode
      low_cmd_.motor_cmd[i].q = PosStopF;
      low_cmd_.motor_cmd[i].kp = 0;
      low_cmd_.motor_cmd[i].dq = VelStopF;
      low_cmd_.motor_cmd[i].kd = 0;
      low_cmd_.motor_cmd[i].tau = 0;
    }
  }

  void LowCmdWrite() {
    // Don't do anything until we've received at least one lowstate
    if (!state_received_) {
      return;
    }

    // Print sensor data at ~2Hz instead of 500Hz
    if ((motiontime_ % 250 == 0) && (percent_4_ < 1)) {
      RCLCPP_INFO(get_logger(), "--- Sensor data ---");
      RCLCPP_INFO(get_logger(), "Joint 0 pos: %.4f",
                  low_state_.motor_state[0].q);
      RCLCPP_INFO(get_logger(), "IMU accel  x: %.4f  y: %.4f  z: %.4f",
                  low_state_.imu_state.accelerometer[0],
                  low_state_.imu_state.accelerometer[1],
                  low_state_.imu_state.accelerometer[2]);
      RCLCPP_INFO(get_logger(), "Foot force[0]: %d",
                  low_state_.foot_force[0]);
    }

    if ((percent_4_ >= 1) && (!done_)) {
      RCLCPP_INFO(get_logger(), "Stand example complete.");
      done_ = true;
    }

    motiontime_++;

    // Wait 1 second (500 ticks at 500Hz) before starting motion
    if (motiontime_ < 500) {
      return;
    }

    // Capture start position on first control tick
    if (first_run_) {
      for (int i = 0; i < 12; i++) {
        start_pos_[i] = low_state_.motor_state[i].q;
      }
      first_run_ = false;
    }

    // --- Phase 1: start_pos -> target_pos_1 ---
    percent_1_ += 1.0f / duration_1_;
    percent_1_ = std::min(percent_1_, 1.0f);

    if (percent_1_ < 1.0f) {
      for (int j = 0; j < 12; j++) {
        low_cmd_.motor_cmd[j].q =
            (1.0f - percent_1_) * start_pos_[j] +
            percent_1_ * target_pos_1_[j];
        low_cmd_.motor_cmd[j].dq = 0;
        low_cmd_.motor_cmd[j].kp = kp_;
        low_cmd_.motor_cmd[j].kd = kd_;
        low_cmd_.motor_cmd[j].tau = 0;
      }
    }

    // --- Phase 2: target_pos_1 -> target_pos_2 ---
    if ((percent_1_ >= 1.0f) && (percent_2_ < 1.0f)) {
      percent_2_ += 1.0f / duration_2_;
      percent_2_ = std::min(percent_2_, 1.0f);

      for (int j = 0; j < 12; j++) {
        low_cmd_.motor_cmd[j].q =
            (1.0f - percent_2_) * target_pos_1_[j] +
            percent_2_ * target_pos_2_[j];
        low_cmd_.motor_cmd[j].dq = 0;
        low_cmd_.motor_cmd[j].kp = kp_;
        low_cmd_.motor_cmd[j].kd = kd_;
        low_cmd_.motor_cmd[j].tau = 0;
      }
    }

    // --- Phase 3: hold target_pos_2 ---
    if ((percent_1_ >= 1.0f) && (percent_2_ >= 1.0f) &&
        (percent_3_ < 1.0f)) {
      percent_3_ += 1.0f / duration_3_;
      percent_3_ = std::min(percent_3_, 1.0f);

      for (int j = 0; j < 12; j++) {
        low_cmd_.motor_cmd[j].q = target_pos_2_[j];
        low_cmd_.motor_cmd[j].dq = 0;
        low_cmd_.motor_cmd[j].kp = kp_;
        low_cmd_.motor_cmd[j].kd = kd_;
        low_cmd_.motor_cmd[j].tau = 0;
      }
    }

    // --- Phase 4: target_pos_2 -> target_pos_3 ---
    if ((percent_1_ >= 1.0f) && (percent_2_ >= 1.0f) &&
        (percent_3_ >= 1.0f) && (percent_4_ <= 1.0f)) {
      percent_4_ += 1.0f / duration_4_;
      percent_4_ = std::min(percent_4_, 1.0f);

      for (int j = 0; j < 12; j++) {
        low_cmd_.motor_cmd[j].q =
            (1.0f - percent_4_) * target_pos_2_[j] +
            percent_4_ * target_pos_3_[j];
        low_cmd_.motor_cmd[j].dq = 0;
        low_cmd_.motor_cmd[j].kp = kp_;
        low_cmd_.motor_cmd[j].kd = kd_;
        low_cmd_.motor_cmd[j].tau = 0;
      }
    }

    get_crc(low_cmd_);
    low_cmd_pub_->publish(low_cmd_);
  }

  // --- Members ---
  float kp_ = 60.0f;
  float kd_ = 5.0f;
  int motiontime_ = 0;

  unitree_go::msg::LowCmd low_cmd_;
  unitree_go::msg::LowState low_state_;

  rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr low_cmd_pub_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr low_state_sub_;
  rclcpp::CallbackGroup::SharedPtr state_cbg_;
  std::thread cmd_thread_;

  bool state_received_ = false;
  bool first_run_ = true;
  bool done_ = false;

  // Target joint positions for each phase
  float start_pos_[12]{};

  float target_pos_1_[12] = {0.0f,  1.36f, -2.65f, 0.0f, 1.36f, -2.65f,
                             -0.2f, 1.36f, -2.65f, 0.2f, 1.36f, -2.65f};

  float target_pos_2_[12] = {0.0f, 0.67f, -1.3f, 0.0f, 0.67f, -1.3f,
                             0.0f, 0.67f, -1.3f, 0.0f, 0.67f, -1.3f};

  float target_pos_3_[12] = {-0.35f, 1.36f, -2.65f, 0.35f, 1.36f, -2.65f,
                             -0.5f,  1.36f, -2.65f, 0.5f,  1.36f, -2.65f};

  // Duration of each phase in ticks (at 500Hz)
  float duration_1_ = 500.0f;   // 1.0s
  float duration_2_ = 500.0f;   // 1.0s
  float duration_3_ = 1000.0f;  // 2.0s
  float duration_4_ = 900.0f;   // 1.8s

  float percent_1_ = 0.0f;
  float percent_2_ = 0.0f;
  float percent_3_ = 0.0f;
  float percent_4_ = 0.0f;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<LowLevelCmdNode>();
  node->StartCmdLoop();

  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 2);
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}