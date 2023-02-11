#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

gpuci_conda_retry remove --force pyarrow arrow-cpp openssl librdkafka
rapids-mamba-retry install -y "pyarrow=11" "libarrow=11" "librdkafka=1.7.0"

rapids-print-env

rapids-logger "Begin cpp build"

rapids-mamba-retry mambabuild conda/recipes/libcudf

rapids-upload-conda-to-s3 cpp
