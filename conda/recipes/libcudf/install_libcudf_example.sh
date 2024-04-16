#!/bin/bash
# Copyright (c) 2018-2024, NVIDIA CORPORATION.

# store old CMAKE_INSTALL_PREFIX if set
CMAKE_INSTALL_PREFIX_OLD=${CMAKE_INSTALL_PREFIX}

# set CMAKE_INSTALL_PREFIX to $CONDA_PREFIX
export CMAKE_INSTALL_PREFIX=${CONDA_PREFIX:-/usr}

# build and install libcudf examples
. ./cpp/examples/build.sh

# set CMAKE_INSTALL_PREFIX to the original one
export CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX_OLD}