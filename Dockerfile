FROM ros:humble

RUN apt update && apt upgrade -y

RUN apt install ros-humble-desktop -y

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
    ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "arm64" ]; then ORT_ARCH="aarch64"; else ORT_ARCH="x64"; fi && \
    ORT_TGZ="onnxruntime-linux-${ORT_ARCH}-1.19.2.tgz" && \
    wget -q "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/${ORT_TGZ}" && \
    tar xf "${ORT_TGZ}" && \
    ORT_DIR=$(tar tf "${ORT_TGZ}" | head -1 | cut -d'/' -f1) && \
    cp -r "${ORT_DIR}/include/"* /usr/local/include/ && \
    cp -r "${ORT_DIR}/lib/"* /usr/local/lib/ && \
    ldconfig && \
    rm -rf "${ORT_DIR}" "${ORT_TGZ}"

# Install unitree_sdk2_python
RUN CYCLONEDDS_HOME=/usr/local pip3 install --no-deps /workspace/dependencies/unitree_sdk2_python

RUN mkdir -p /workspace/ros2_ws/src

WORKDIR /workspace/ros2_ws

SHELL ["/bin/bash", "-c"]

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
