#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

RAPIDS_VERSION="$(rapids-version)"

rapids-logger "Generate C++ testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-print-env

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  "libcudf=${RAPIDS_VERSION}" \
  "libcudf_kafka=${RAPIDS_VERSION}" \
  "libcudf-tests=${RAPIDS_VERSION}" \
  "libcudf-example=${RAPIDS_VERSION}"

rapids-logger "Check GPU usage"
nvidia-smi
