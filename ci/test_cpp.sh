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

# Get library for finding incorrect default stream usage.
STREAM_IDENTIFY_LIB="${CONDA_PREFIX}/lib/libcudf_identify_stream_usage.so"

echo "STREAM_IDENTIFY_LIB=${STREAM_IDENTIFY_LIB}"

# Run libcudf and libcudf_kafka gtests from libcudf-tests package
rapids-logger "Run gtests"

# TODO: exit code handling is too verbose. Find a cleaner solution.

for gt in "$CONDA_PREFIX"/bin/gtests/{libcudf,libcudf_kafka}/* ; do
    test_name=$(basename ${gt})
    echo "Running gtest $test_name"

    # TODO: This strategy for using the stream lib will need to change when we
    # switch to invoking ctest. For one, we will want to set the test
    # properties to use the lib (which means that the decision will be made at
    # CMake-configure time instead of runtime). We may also need to leverage
    # something like gtest_discover_tests to be able to filter on the
    # underlying test names.
    if [[ ${test_name} == "SPAN_TEST" ]]; then
        # This one test is specifically designed to test using a thrust device
        # vector, so we expect and allow it to include default stream usage.
        gtest_filter="SpanTest.CanConstructFromDeviceContainers"
        GTEST_CUDF_STREAM_MODE="custom" LD_PRELOAD=${STREAM_IDENTIFY_LIB} ${gt} --gtest_output=xml:${RAPIDS_TESTS_DIR} --gtest_filter="-${gtest_filter}" && \
            ${gt} --gtest_output=xml:${RAPIDS_TESTS_DIR} --gtest_filter="${gtest_filter}"
    else
        GTEST_CUDF_STREAM_MODE="custom" LD_PRELOAD=${STREAM_IDENTIFY_LIB} ${gt} --gtest_output=xml:${RAPIDS_TESTS_DIR}
    fi
done

if [[ "${RAPIDS_BUILD_TYPE}" == "nightly" ]]; then
    rapids-logger "Memcheck gtests with rmm_mode=cuda"
    export GTEST_CUDF_RMM_MODE=cuda
    COMPUTE_SANITIZER_CMD="compute-sanitizer --tool memcheck"
    for gt in "$CONDA_PREFIX"/bin/gtests/{libcudf,libcudf_kafka}/* ; do
        test_name=$(basename ${gt})
        if [[ "$test_name" == "ERROR_TEST" ]]; then
            continue
        fi
        echo "Running gtest $test_name"
        ${COMPUTE_SANITIZER_CMD} ${gt} | tee "${RAPIDS_TESTS_DIR}${test_name}.cs.log"
    done
    unset GTEST_CUDF_RMM_MODE
    # TODO: test-results/*.cs.log are processed in gpuci
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
