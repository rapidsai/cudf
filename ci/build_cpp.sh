#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

# With boa installed conda build forward to boa
rapids-conda-retry mambabuild \
    conda/recipes/libcudf

rapids-upload-conda-to-s3 cpp
