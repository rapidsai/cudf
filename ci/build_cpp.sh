#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

# Use gha-tools fork
wget https://github.com/bdice/gha-tools/archive/refs/heads/wheel-python-pure.zip
unzip wheel-python-pure.zip
cp gha-tools-wheel-python-pure/tools/* /usr/local/bin

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

version=$(rapids-generate-version)

rapids-logger "Begin cpp build"

# With boa installed conda build forward to boa
RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
    conda/recipes/libcudf

rapids-upload-conda-to-s3 cpp
