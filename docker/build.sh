#!/bin/bash
cd $(dirname "${BASH_SOURCE[0]}")

source params.sh

docker build -t "${IMAGENAME}" .
