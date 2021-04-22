#!/bin/bash

# Copyright (c) 2019-2021, NVIDIA CORPORATION.

# libcudf examples build script

# Add libcudf examples build scripts down below

################################################################################
# Basic example

# Configure
cmake -S $REPODIR/cpp/example/basic -B $REPODIR/cpp/example/basic/build
# Build
cd ${$REPODIR}/cpp/example/basic/build
cmake --build . -j${PARALLEL_LEVEL} ${VERBOSE_FLAG}
