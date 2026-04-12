#!/bin/bash
source /opt/ros/humble/setup.bash

cd /workspace/ros2_ws
rosdep install --from-path src --ignore-src -r -y
colcon build --symlink-install --parallel-workers $(( $(nproc) / 2 ))
source /workspace/ros2_ws/install/setup.bash

echo "source /workspace/ros2_ws/install/setup.bash" >> ~/.bashrc
echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc
echo "export CYCLONEDDS_URI=file:///workspace/dependencies/cyclonedds.xml" >> ~/.bashrc
echo "alias mujoco='export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu'" >> ~/.bashrc

cd /workspace/ros2_ws/

exec bash
