#!/bin/bash
# Copyright (c) 2018-2023, NVIDIA CORPORATION.

export cudf_ROOT="$(realpath ./cpp/build)"

[[ $(arch) == "x86_64" ]] && targetsDir="targets/x86_64-linux"
[[ $(arch) == "aarch64" ]] && targetsDir="targets/sbsa-linux"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:${BUILD_PREFIX}/${targetsDir}/lib:${BUILD_PREFIX}/${targetsDir}/lib/stubs"

./build.sh -n -v libcudf libcudf_kafka benchmarks tests --build_metrics --incl_cache_stats --cmake-args=\"-DCMAKE_INSTALL_LIBDIR=lib -DCUDF_ENABLE_ARROW_S3=ON\"
