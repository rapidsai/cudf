#!/bin/bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.

export cudf_ROOT="$(realpath ./cpp/build)"
./build.sh -n -v libcudf libcudf_kafka benchmarks tests --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib\"
