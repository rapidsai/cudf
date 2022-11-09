#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh
conda activate base

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test
conda activate test

rapids-print-env

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-mamba-retry install \
  -c "${CPP_CHANNEL}" \
  libcudf libcudf_kafka libcudf-tests

rapids-logger "Check GPU usage"
nvidia-smi

set +e

# Set up library for finding incorrect default stream usage.
pushd "cpp/tests/utilities/identify_stream_usage/"
mkdir build && cd build && cmake .. -GNinja && ninja && ninja test
STREAM_IDENTIFY_LIB="$(realpath build/libidentify_stream_usage.so)"
echo "STREAM_IDENTIFY_LIB=${STREAM_IDENTIFY_LIB}"
popd

# Run libcudf and libcudf_kafka gtests from libcudf-tests package
rapids-logger "Run gtests"
TESTRESULTS_DIR=test-results
mkdir -p ${TESTRESULTS_DIR}
SUITEERROR=0

# TODO: exit code handling is too verbose. Find a cleaner solution.

for gt in "$CONDA_PREFIX"/bin/gtests/{libcudf,libcudf_kafka}/* ; do
    test_name=$(basename ${gt})
    echo "Running gtest $test_name"
    if [[ ${test_name} == "SPAN_TEST" ]]; then
        # This one test is specifically designed to test using a thrust device
        # vector, so we expect and allow it to include default stream usage.
        gtest_filter="SpanTest.CanConstructFromDeviceContainers"
        GTEST_CUDF_STREAM_MODE="custom" LD_PRELOAD=${STREAM_IDENTIFY_LIB} ${gt} --gtest_output=xml:${TESTRESULTS_DIR} --gtest_filter="-${gtest_filter}" && \
            ${gt} --gtest_output=xml:${TESTRESULTS_DIR} --gtest_filter="${gtest_filter}"
    else
        GTEST_CUDF_STREAM_MODE="custom" LD_PRELOAD=${STREAM_IDENTIFY_LIB} ${gt} --gtest_output=xml:${TESTRESULTS_DIR}
    fi

    exitcode=$?
    if (( ${exitcode} != 0 )); then
        SUITEERROR=${exitcode}
        echo "FAILED: GTest ${gt}"
    fi
done

rapids-logger "Run gtests with kvikio"
# Test libcudf (csv, orc, and parquet) with `LIBCUDF_CUFILE_POLICY=KVIKIO`
for test_name in "CSV_TEST" "ORC_TEST" "PARQUET_TEST"; do
    gt="$CONDA_PREFIX/bin/gtests/libcudf/${test_name}"
    echo "Running gtest $test_name (LIBCUDF_CUFILE_POLICY=KVIKIO)"
    LIBCUDF_CUFILE_POLICY=KVIKIO ${gt} --gtest_output=xml:${TESTRESULTS_DIR}/
    exitcode=$?
    if (( ${exitcode} != 0 )); then
        SUITEERROR=${exitcode}
        echo "FAILED: GTest ${gt}"
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
        ${COMPUTE_SANITIZER_CMD} ${gt} | tee "${TESTRESULTS_DIR}/${test_name}.cs.log"
    done
    unset GTEST_CUDF_RMM_MODE
    # TODO: test-results/*.cs.log are processed in gpuci
fi

exit ${SUITEERROR}
