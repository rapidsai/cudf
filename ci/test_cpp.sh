#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh
conda activate base

rapids-dependency-file-generator \
  --output conda \
  --file_key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*}" > env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test
conda activate test

rapids-print-env

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-mamba-retry install \
  -c "${CPP_CHANNEL}" \
  libcudf libcudf_kafka libcudf-tests

TESTRESULTS_DIR=test-results
mkdir -p ${TESTRESULTS_DIR}
SUITEERROR=0

rapids-logger "Check GPU usage"
nvidia-smi

set +e

# TODO: Insert trap to remove jitify cache from ci/gpu/build.sh

rapids-logger "Running googletests"
for gt in "$CONDA_PREFIX/bin/gtests/libcudf/"* ; do
    test_name=$(basename ${gt})
    if [[ ${test_name} == "SPAN_TEST" ]]; then
        # This one test is specifically designed to test using a thrust device
        # vector, so we expect and allow it to include default stream usage.
        gtest_filter="SpanTest.CanConstructFromDeviceContainers"
        GTEST_CUDF_STREAM_MODE="custom" LD_PRELOAD=${STREAM_IDENTIFY_LIB} ${gt} --gtest_output=xml:${TESTRESULTS_DIR} --gtest_filter="-${gtest_filter}"
        ${gt} --gtest_output=xml:"$WORKSPACE/test-results/" --gtest_filter="${gtest_filter}"
    else
        GTEST_CUDF_STREAM_MODE="custom" LD_PRELOAD=${STREAM_IDENTIFY_LIB} ${gt} --gtest_output=xml:${TESTRESULTS_DIR}/
    fi

    exitcode=$?
    if (( ${exitcode} != 0 )); then
        SUITEERROR=${exitcode}
        echo "FAILED: GTest ${gt}"
    fi
done

# Test libcudf (csv, orc, and parquet) with `LIBCUDF_CUFILE_POLICY=KVIKIO`
for test_name in "CSV_TEST" "ORC_TEST" "PARQUET_TEST"; do
    gt="$CONDA_PREFIX/bin/gtests/libcudf/${test_name}"
    echo "Running GoogleTest $test_name (LIBCUDF_CUFILE_POLICY=KVIKIO)"
    LIBCUDF_CUFILE_POLICY=KVIKIO ${gt} --gtest_output=xml:${TESTRESULTS_DIR}/
    exitcode=$?
    if (( ${exitcode} != 0 )); then
        SUITEERROR=${exitcode}
        echo "FAILED: GTest ${gt}"
    fi
done

exit ${SUITEERROR}
