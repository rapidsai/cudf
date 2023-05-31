#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

_CUDA_MAJOR=$(nvcc --version | tail -n 2 | head -n 1 | cut -d',' -f 2 | cut -d' ' -f 3 | cut -d'.' -f 1)

LIBRMM_CHANNEL=$(rapids-get-artifact ci/rmm/pull-request/1278/4e1392d/rmm_conda_cpp_cuda${CUDA_MAJOR}_$(arch).tar.gz)

rapids-mamba-retry mambabuild  --channel "${LIBRMM_CHANNEL}" conda/recipes/libcudf

rapids-upload-conda-to-s3 cpp
