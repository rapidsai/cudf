#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Generate C++ testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key all \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n test_streaming

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test_streaming
set -u

source rapids-configure-sccache

rapidps-logger "Start sccache server"

sccache --start-server
sccache --show-adv-stats
sccache --dist-status

rapids-print-env

rapids-logger "Run C++ build"

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
export RAPIDS_CUDA_MAJOR

cmake -S cpp -B cpp/build -GNinja \
  -DCUDA_STATIC_RUNTIME=OFF \
  -DCUDF_BUILD_STREAMS_TEST_UTIL=ON \
  -DBUILD_SHARED_LIBS=ON
mkdir cpp/install
cmake --build cpp/build "-j${PARALLEL_LEVEL}" -v
cmake --install cpp/build --prefix cpp/install
cmake --install cpp/build --prefix cpp/install --component testing

sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

cat "$SCCACHE_ERROR_LOG"
