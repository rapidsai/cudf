#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1374 cpp)

version=$(rapids-generate-version)

rapids-logger "Begin cpp build"

# With boa installed conda build forward to boa
RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
    --channel "${LIBRMM_CHANNEL}" \
    conda/recipes/libcudf

rapids-upload-conda-to-s3 cpp
