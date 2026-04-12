// Unitree Go2 LowCmd CRC32 computation.
// Verbatim from unitree_ros2_examples/include/common/motor_crc.h.
// Copyright (c) 2020, Unitree Robotics. All rights reserved.

#ifndef GO2_VELOCITY_MOTOR_CRC_HPP_
#define GO2_VELOCITY_MOTOR_CRC_HPP_

#include <stdint.h>

#include <array>

#include "unitree_go/msg/low_cmd.hpp"

constexpr int HIGHLEVEL = 0xee;
constexpr int LOWLEVEL = 0xff;
constexpr int TRIGERLEVEL = 0xf0;
constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

typedef struct {
  uint8_t off;
  std::array<uint8_t, 3> reserve;
} BmsCmd;

typedef struct {
  uint8_t mode;
  float q;
  float dq;
  float tau;
  float Kp;
  float Kd;
  std::array<uint32_t, 3> reserve;
} MotorCmd;

typedef struct {
  std::array<uint8_t, 2> head;
  uint8_t levelFlag;
  uint8_t frameReserve;

  std::array<uint32_t, 2> SN;
  std::array<uint32_t, 2> version;
  uint16_t bandWidth;
  std::array<MotorCmd, 20> motorCmd;
  BmsCmd bms;
  std::array<uint8_t, 40> wirelessRemote;
  std::array<uint8_t, 12> led;
  std::array<uint8_t, 2> fan;
  uint8_t gpio;
  uint32_t reserve;

  uint32_t crc;
} LowCmd;

uint32_t crc32_core(uint32_t* ptr, uint32_t len);
void get_crc(unitree_go::msg::LowCmd& msg);

#endif  // GO2_VELOCITY_MOTOR_CRC_HPP_
