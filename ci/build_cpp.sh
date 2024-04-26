#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

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
