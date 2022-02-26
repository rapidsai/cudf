#!/bin/bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.

# ./build.sh -n -v libcudf libcudf_kafka tests benchmarks --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib\"
# cmake --install build --component testing

echo "pwd: $PWD"
echo "!?cpp/build"
ls -l cpp/build

echo "!?cpp/libcudf_kafka/build"
ls -l cpp/libcudf_kafka/build
