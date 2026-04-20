#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly artifact_name="$1"

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Download stream test artifacts"
STREAM_TESTS="$(rapids-download-from-github "$artifact_name")"

rapids-logger "Generate C++ testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key stream_tests \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n stream_tests

# Temporarily allow unbound variables for conda activation.
set +u
conda activate stream_tests
set -u

rapids-print-env

rapids-logger "Run C++ tests"

ctest --test-dir "${STREAM_TESTS}/bin/gtests/libcudf" --output-on-failure
