#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
LIBRMM_CHANNEL=$(rapids-get-artifact ci/rmm/pull-request/1223/9a74395/rmm_conda_cpp_cuda${RAPIDS_CUDA_MAJOR}_$(arch).tar.gz)

rapids-print-env

rapids-logger "Begin cpp build"

rapids-mamba-retry mambabuild \
    --channel "${LIBRMM_CHANNEL}" \
    conda/recipes/libcudf

rapids-upload-conda-to-s3 cpp
