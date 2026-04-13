cmake_minimum_required(VERSION 3.8)
project(go2_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(Threads REQUIRED)
find_package(unitree_sdk2 REQUIRED)
find_package(unitree_api REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/Go2SportMode.srv"
  "srv/Go2Damp.srv"
)

add_executable(go2_sport_mode_service src/go2_sport_mode_service.cpp)
ament_target_dependencies(go2_sport_mode_service rclcpp)
target_link_libraries(go2_sport_mode_service unitree_sdk2)

add_executable(go2_damp_service src/go2_damp_service.cpp)
ament_target_dependencies(go2_damp_service rclcpp unitree_api)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(go2_sport_mode_service "${cpp_typesupport_target}")
target_link_libraries(go2_damp_service "${cpp_typesupport_target}")

install(TARGETS
  go2_sport_mode_service
  go2_damp_service
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
