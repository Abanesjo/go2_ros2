cmake_minimum_required(VERSION 3.14)
project(go2_bridge)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(unitree_go REQUIRED)

add_executable(go2_bridge_node
  src/go2_bridge_node.cpp
  src/motor_crc.cpp)
ament_target_dependencies(go2_bridge_node
  rclcpp sensor_msgs std_msgs unitree_go)

install(TARGETS go2_bridge_node DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
