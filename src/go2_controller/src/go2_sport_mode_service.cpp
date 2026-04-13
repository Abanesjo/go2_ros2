#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>

#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>

#include "go2_controller/srv/go2_sport_mode.hpp"

using unitree::robot::ChannelFactory;
using unitree::robot::b2::MotionSwitcherClient;

namespace
{
std::string GetServiceName(const std::string& form, const std::string& name)
{
  if (form == "0") {
    if (name == "normal") {
      return "sport_mode";
    }
    if (name == "ai") {
      return "ai_sport";
    }
    if (name == "advanced") {
      return "advanced_sport";
    }
  } else {
    if (name == "ai-w") {
      return "wheeled_sport(go2W)";
    }
    if (name == "normal-w") {
      return "wheeled_sport(b2W)";
    }
  }
  return name;
}
}

class Go2SportModeService final : public rclcpp::Node
{
public:
  Go2SportModeService()
  : rclcpp::Node("go2_sport_mode_service")
  {
    const std::string iface = this->declare_parameter<std::string>("network_interface", "enp130s0");
    ChannelFactory::Instance()->Init(0, iface);

    msc_.SetTimeout(10.0f);
    msc_.Init();

    service_ = this->create_service<go2_controller::srv::Go2SportMode>(
      "go2_sport_mode",
      std::bind(
        &Go2SportModeService::HandleRequest,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Go2 sport mode service ready on interface '%s'", iface.c_str());
  }

private:
  void HandleRequest(
    const std::shared_ptr<go2_controller::srv::Go2SportMode::Request> request,
    std::shared_ptr<go2_controller::srv::Go2SportMode::Response> response)
  {
    std::string command = request->command;
    if (command.empty()) {
      command = "status";
    }

    if (command != "status" && command != "on" && command != "off" && command != "toggle") {
      response->success = false;
      response->message = "Unknown command: " + command;
      return;
    }

    std::string robot_form;
    std::string motion_name;
    int32_t ret = msc_.CheckMode(robot_form, motion_name);
    if (ret != 0) {
      response->success = false;
      response->message = "Failed to check mode. Error code: " + std::to_string(ret);
      return;
    }

    bool is_active = !motion_name.empty();
    std::string service_name = is_active ? GetServiceName(robot_form, motion_name) : "none";

    if (command == "status") {
      response->success = true;
      response->message = is_active ? "Sport mode is ON" : "Sport mode is OFF";
      response->is_active = is_active;
      response->robot_form = robot_form;
      response->motion_name = motion_name;
      response->service_name = service_name;
      return;
    }

    if (command == "toggle") {
      command = is_active ? "off" : "on";
    }

    if (command == "off") {
      if (!is_active) {
        response->success = true;
        response->message = "Sport mode is already OFF";
        response->is_active = is_active;
        response->robot_form = robot_form;
        response->motion_name = motion_name;
        response->service_name = service_name;
        return;
      }

      ret = msc_.ReleaseMode();
      if (ret != 0) {
        response->success = false;
        response->message = "Failed to disable sport mode. Error code: " + std::to_string(ret);
        response->is_active = is_active;
        response->robot_form = robot_form;
        response->motion_name = motion_name;
        response->service_name = service_name;
        return;
      }
    }

    if (command == "on") {
      if (is_active) {
        response->success = true;
        response->message = "Sport mode is already ON";
        response->is_active = is_active;
        response->robot_form = robot_form;
        response->motion_name = motion_name;
        response->service_name = service_name;
        return;
      }

      ret = msc_.SelectMode("normal");
      if (ret != 0) {
        response->success = false;
        response->message = "Failed to enable sport mode. Error code: " + std::to_string(ret);
        response->is_active = is_active;
        response->robot_form = robot_form;
        response->motion_name = motion_name;
        response->service_name = service_name;
        return;
      }
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
    ret = msc_.CheckMode(robot_form, motion_name);
    if (ret != 0) {
      response->success = false;
      response->message = "Mode changed but failed to read back. Error code: " + std::to_string(ret);
      return;
    }

    is_active = !motion_name.empty();
    service_name = is_active ? GetServiceName(robot_form, motion_name) : "none";

    response->success = true;
    response->message = is_active ? "Sport mode is ON" : "Sport mode is OFF";
    response->is_active = is_active;
    response->robot_form = robot_form;
    response->motion_name = motion_name;
    response->service_name = service_name;
  }

  MotionSwitcherClient msc_;
  rclcpp::Service<go2_controller::srv::Go2SportMode>::SharedPtr service_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Go2SportModeService>());
  rclcpp::shutdown();
  return 0;
}
