#!/bin/bash
IMAGE_NAME="go2_velocity"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

xhost +local:docker

docker run -it --rm --name "$IMAGE_NAME" --network host \
  --gpus all \
  --runtime=nvidia \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$SCRIPT_DIR/src:/workspace/ros2_ws/src" \
  "$IMAGE_NAME"
