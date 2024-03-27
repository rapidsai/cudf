#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

# Common setup steps shared by Python test jobs

set -euo pipefail

DISTRIBUTION=${1}

RESULTS_DIR=${RAPIDS_TESTS_DIR:-"$(mktemp -d)"}
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${RESULTS_DIR}/test-results"}/
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${RESULTS_DIR}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

if [[ ${DISTRIBUTION} == "conda" ]]; then
  . /opt/conda/etc/profile.d/conda.sh

  rapids-logger "Generate Python testing dependencies"

  ENV_YAML_DIR="$(mktemp -d)"

  rapids-dependency-file-generator \
    --output conda \
    --file_key test_python \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee "${ENV_YAML_DIR}/env.yaml"

  rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n test

  # Temporarily allow unbound variables for conda activation.
  set +u
  conda activate test
  set -u

  rapids-logger "Downloading artifacts from previous jobs"
  CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
  PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

  rapids-print-env

  rapids-mamba-retry install \
    --channel "${CPP_CHANNEL}" \
    --channel "${PYTHON_CHANNEL}" \
    cudf libcudf
elif [[ ${DISTRIBUTION} == "wheel" ]]; then
  RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
  RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

  # echo to expand wildcard before adding `[extra]` requires for pip
  pip install $(echo ./dist/cudf*.whl)[test]
else
  exit 1
fi
