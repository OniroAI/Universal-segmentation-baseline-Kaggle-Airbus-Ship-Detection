#!/bin/bash
cd $(dirname "${BASH_SOURCE[0]}")

NAME="kaggle_ships"
IMAGENAME="${NAME}"
CONTNAME="--name=${NAME}"
NET="--net=host"
IPC="--ipc=host"

X11_VOLUMES=""
if [ -e "/tmp/.X11-unix" ]; then
  X11_VOLUMES="-v /tmp/.X11-unix:/tmp/.X11-unix -v ${HOME}/.Xauthority:/root/.Xauthority"
  DISPLAY="-e DISPLAY=$DISPLAY"
fi

VOLUMES="-v $(pwd)/..:/workdir"

if [ -e "personal_params.sh" ]; then
  source personal_params.sh
fi
