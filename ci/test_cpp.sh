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

# TODO: Disabling stream identification for now.
# Set up library for finding incorrect default stream usage.
#pushd "cpp/tests/utilities/identify_stream_usage/"
#mkdir build && cd build && cmake .. -GNinja && ninja && ninja test
#STREAM_IDENTIFY_LIB="$(realpath build/libidentify_stream_usage.so)"
#echo "STREAM_IDENTIFY_LIB=${STREAM_IDENTIFY_LIB}"
#popd

# Run libcudf and libcudf_kafka gtests from libcudf-tests package
rapids-logger "Run gtests"

# TODO: exit code handling is too verbose. Find a cleaner solution.

for gt in "$CONDA_PREFIX"/bin/gtests/{libcudf,libcudf_kafka}/* ; do
    test_name=$(basename ${gt})
    echo "Running gtest $test_name"
    ${gt} --gtest_output=xml:${RAPIDS_TESTS_DIR}
    # TODO: Disabling stream identification for now.
    #if [[ ${test_name} == "SPAN_TEST" ]]; then
    #    # This one test is specifically designed to test using a thrust device
    #    # vector, so we expect and allow it to include default stream usage.
    #    gtest_filter="SpanTest.CanConstructFromDeviceContainers"
    #    GTEST_CUDF_STREAM_MODE="custom" LD_PRELOAD=${STREAM_IDENTIFY_LIB} ${gt} --gtest_output=xml:${RAPIDS_TESTS_DIR} --gtest_filter="-${gtest_filter}" && \
    #        ${gt} --gtest_output=xml:${RAPIDS_TESTS_DIR} --gtest_filter="${gtest_filter}"
    #else
    #    GTEST_CUDF_STREAM_MODE="custom" LD_PRELOAD=${STREAM_IDENTIFY_LIB} ${gt} --gtest_output=xml:${RAPIDS_TESTS_DIR}
    #fi
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
