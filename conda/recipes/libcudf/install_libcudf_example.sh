#!/bin/bash
# Copyright (c) 2018-2024, NVIDIA CORPORATION.

export CMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
./cpp/examples/build.sh
