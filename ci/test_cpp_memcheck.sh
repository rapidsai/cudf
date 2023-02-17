#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-print-env

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  libcudf libcudf_kafka libcudf-tests

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run gtests with compute-sanitizer
echo "RAPIDS_BUILD_TYPE=${RAPIDS_BUILD_TYPE}"

if [[ "${RAPIDS_BUILD_TYPE}" == "nightly" ]]; then
    rapids-logger "Memcheck gtests with rmm_mode=cuda"
    export GTEST_CUDF_RMM_MODE=cuda
    COMPUTE_SANITIZER_CMD="compute-sanitizer --tool memcheck"
    for gt in "$CONDA_PREFIX"/bin/gtests/{libcudf,libcudf_kafka}/* ; do
        test_name=$(basename ${gt})
        if [[ "$test_name" == "ERROR_TEST" ]]; then
            continue
        fi
        if [[ "$test_name" == "STREAM_IDENTIFICATION_TEST" ]]; then
            continue
        fi
        echo "Running gtest $test_name"
        ${COMPUTE_SANITIZER_CMD} ${gt} --gtest_output=xml:"${RAPIDS_TESTS_DIR}${test_name}.xml"
    done
    unset GTEST_CUDF_RMM_MODE
    # TODO: test-results/*.cs.log are processed in CI
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
