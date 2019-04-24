FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm cmake unzip git wget \
    sysstat libtcmalloc-minimal4 autoconf libtool flex bison \
    pkg-config glew-utils libglew-dev libglew2.0 \
    freeglut3 freeglut3-dev libghc-glut-dev \
    libghc-glut-doc libghc-glut-prof libalut-dev libxmu-dev libxmu-headers libxmu6 \
    libxmu6-dbg libxmuu-dev libxmuu1 libxmuu1-dbg libxrender1 \
    libturbojpeg \
    python3 python3-pip python3-dev python3-setuptools &&\
    echo 'alias python=python3' >> ~/.bashrc &&\
    echo 'alias pip=pip3' >> ~/.bashrc &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility

RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl &&\
    pip3 install torchvision==0.2.2 &&\
    rm -rf ~/.cache/pip

RUN pip3 install \
    easydict==1.7 \
    numpy \
    pandas \
    pytorch-argus==0.0.8 \
    opencv-python==3.4.2.17 \
    scipy==1.2.1 \
    matplotlib==3.0.3 \
    jpeg4py==0.1.4 \
    transitions==0.6.9 \
    jupyter \
    scikit-image \
    cython

RUN pip3 install pycocotools

ENV PYTHONPATH $PYTHONPATH:/workdir
ENV TORCH_HOME=/workdir/data/.torch
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/opt/conda/lib/

WORKDIR /workdir
