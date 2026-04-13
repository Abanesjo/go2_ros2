FROM osrf/ros:humble-desktop-full

RUN apt update && apt upgrade -y

# ROS2 and build deps
RUN apt install -y \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-generator-dds-idl \
    ros-humble-rosbag2-cpp \
    libyaml-cpp-dev \
    libeigen3-dev \
    python3-pip \
    libboost-all-dev \
    libspdlog-dev \
    libfmt-dev\ 
    tmux

# Pin numpy/scipy/opencv before installing unitree_sdk2_python (which pulls numpy2 + opencv4 otherwise)
RUN pip3 install numpy==1.26.4 scipy==1.13.1 opencv-contrib-python==4.7.0.72

COPY dependencies /workspace/dependencies

# Build & install CycloneDDS
RUN cd /workspace/dependencies/cyclonedds && mkdir build && cd build && \
    cmake .. && make -j$(($(nproc) / 2)) && make install && ldconfig

# Build & install unitree_sdk2
RUN cd /workspace/dependencies/unitree_sdk2 && mkdir build && cd build && \
    cmake .. && make -j$(($(nproc) / 2)) && make install && ldconfig

# Install ONNX Runtime C++ (for rl_controller_node) — v1.19.2 supports IR version 10
RUN apt install -y wget && \
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz && \
    tar xf onnxruntime-linux-x64-1.19.2.tgz && \
    cp -r onnxruntime-linux-x64-1.19.2/include/* /usr/local/include/ && \
    cp -r onnxruntime-linux-x64-1.19.2/lib/* /usr/local/lib/ && \
    ldconfig && \
    rm -rf onnxruntime-linux-x64-1.19.2*

# Install unitree_sdk2_python
RUN CYCLONEDDS_HOME=/usr/local pip3 install --no-deps /workspace/dependencies/unitree_sdk2_python

RUN mkdir -p /workspace/ros2_ws/src

WORKDIR /workspace/ros2_ws

SHELL ["/bin/bash", "-c"]

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
