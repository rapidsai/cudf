#!/bin/bash
#
# Copyright (c) 2018, NVIDIA CORPORATION.
set -e

if [ "$BUILD_LIBGDF" == "1" -o "$BUILD_CFFI" == "1" ]; then
    # install libboost
    sudo apt-get update -q
    sudo apt-get install -y libboost-all-dev
    # install libcuda
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/ /" | sudo tee -a /etc/apt/sources.list.d/cuda.list
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/7fa2af80.pub
    sudo apt-get update -q
    sudo apt-get install -y --no-install-recommends cuda-drivers=396.44-1 libcuda1-396
    # install gcc-5
    echo "deb http://archive.ubuntu.com/ubuntu/ xenial main restricted" | sudo tee -a /etc/apt/sources.list
    echo "deb http://archive.ubuntu.com/ubuntu/ xenial-updates main restricted" | sudo tee -a /etc/apt/sources.list
    echo "deb http://security.ubuntu.com/ubuntu/ xenial-security main restricted" | sudo tee -a /etc/apt/sources.list
    sudo apt-get update -q
    sudo apt-get install -y gcc-5 g++-5 cpp-5 libisl15 libmpfr4 libstdc++-5-dev libgcc-5-dev libc6-dev
    # set gcc/g++ paths
    export CC=/usr/bin/gcc-5
    export CXX=/usr/bin/g++-5
    # install cuda
    source ./travisci/libgdf/install-cuda-trusty.sh
    # check versions
    $CC --version
    $CXX --version
    nvcc --version
fi