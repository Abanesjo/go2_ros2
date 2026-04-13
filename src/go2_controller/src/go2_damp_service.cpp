#include <functional>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "unitree_api/msg/request.hpp"
#include "go2_controller/srv/go2_damp.hpp"

constexpr int32_t SPORT_API_ID_DAMP = 1001;

class Go2DampService final : public rclcpp::Node
{
public:
  Go2DampService()
  : rclcpp::Node("go2_damp_service")
  {
    publisher_ = this->create_publisher<unitree_api::msg::Request>(
      "/api/sport/request", 10);

    service_ = this->create_service<go2_controller::srv::Go2Damp>(
      "go2_damp",
      std::bind(
        &Go2DampService::HandleRequest,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Go2 damp service ready.");
  }

private:
  void HandleRequest(
    const std::shared_ptr<go2_controller::srv::Go2Damp::Request>,
    std::shared_ptr<go2_controller::srv::Go2Damp::Response> response)
  {
    RCLCPP_WARN(this->get_logger(), "Damp mode will power down the motors. The robot will collapse if standing.");

    unitree_api::msg::Request req;
    req.header.identity.api_id = SPORT_API_ID_DAMP;
    publisher_->publish(req);

    response->success = true;
    response->message = "Damp mode activated successfully.";
  }

  rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr publisher_;
  rclcpp::Service<go2_controller::srv::Go2Damp>::SharedPtr service_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Go2DampService>());
  rclcpp::shutdown();
  return 0;
}
