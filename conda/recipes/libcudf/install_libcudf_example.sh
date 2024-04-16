#!/bin/bash
# Copyright (c) 2018-2024, NVIDIA CORPORATION.

export CMAKE_INSTALL_PREFIX=${PREFIX}

# build and install libcudf examples
./cpp/examples/build.sh
