#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

# Common setup steps shared by Python test jobs

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

echo "printRAPIDS_COVERAGE_DIR1: ${RAPIDS_COVERAGE_DIR}"
rapids-print-env
echo "printRAPIDS_COVERAGE_DIR2: ${RAPIDS_COVERAGE_DIR}"

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

TESTRESULTS_DIR="${PWD}/test-results"
mkdir -p "${TESTRESULTS_DIR}"
SUITEERROR=0

echo "printRAPIDS_COVERAGE_DIR3: ${RAPIDS_COVERAGE_DIR}"
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
echo "printRAPIDS_COVERAGE_DIR4: ${RAPIDS_COVERAGE_DIR}"
mkdir -p "${RAPIDS_COVERAGE_DIR}"

rapids-mamba-retry install \
  -c "${CPP_CHANNEL}" \
  -c "${PYTHON_CHANNEL}" \
  cudf libcudf
