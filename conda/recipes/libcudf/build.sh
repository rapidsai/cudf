#!/bin/bash
# Copyright (c) 2018-2023, NVIDIA CORPORATION.

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS} -ccbin ${CXX}" # Needed for CUDA 12 nvidia channel compilers
export cudf_ROOT="$(realpath ./cpp/build)"
./build.sh -n -v libcudf libcudf_kafka benchmarks tests --build_metrics --incl_cache_stats --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib -DCUDF_ENABLE_ARROW_S3=ON\"
