#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

LIBRMM_CHANNEL=$(rapids-get-artifact ci/rmm/pull-request/1241/1fd0b84/rmm_conda_cpp_cuda11_$(arch).tar.gz)

rapids-mamba-retry mambabuild  --channel "${LIBRMM_CHANNEL}" conda/recipes/libcudf

rapids-upload-conda-to-s3 cpp
