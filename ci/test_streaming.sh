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

rapids-print-env

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run C++ tests"

ctest --test-dir cpp/build

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
