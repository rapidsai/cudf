#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

# Common setup steps shared by Python test jobs

LIB=$1

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --config "python/cudf/cudf_pandas_third_party_integration_tests/dependencies.yaml" \
  --output conda \
  --file-key test_${LIB} \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
echo "The working directory is ${PWD}"
mkdir -p "${RAPIDS_TESTS_DIR}"

repo_root=$(git rev-parse --show-toplevel)
TEST_DIR=${repo_root}/python/cudf/cudf_pandas_third_party_integration_tests/tests

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest ${LIB}"

NUM_PROCESSES=8
serial_libraries=(
    "tensorflow"
)
for serial_library in "${serial_libraries[@]}"; do
    if [ "${LIB}" = "${serial_library}" ]; then
        NUM_PROCESSES=1
    fi
done

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR} TEST_DIR=${TEST_DIR} NUM_PROCESSES=${NUM_PROCESSES} python/cudf/cudf_pandas_third_party_integration_tests/ci/ci_run_library_tests.sh ${LIB}

rapids-logger "Test script exiting with value: ${EXITCODE}"
exit ${EXITCODE}
