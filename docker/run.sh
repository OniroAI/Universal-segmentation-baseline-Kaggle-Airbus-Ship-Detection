#!/bin/bash
cd $(dirname "${BASH_SOURCE[0]}")

source params.sh

nvidia-docker run --rm -it \
    ${X11_VOLUMES} \
    ${DISPLAY} \
    ${NET} \
    ${IPC} \
    ${VOLUMES} \
    ${CONTNAME} \
    ${IMAGENAME}  \
    bash
