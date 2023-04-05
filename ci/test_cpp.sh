#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

source "$(dirname "$0")/test_cpp_common.sh"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Get library for finding incorrect default stream usage.
STREAM_IDENTIFY_LIB_MODE_CUDF="${CONDA_PREFIX}/lib/libcudf_identify_stream_usage_mode_cudf.so"
STREAM_IDENTIFY_LIB_MODE_TESTING="${CONDA_PREFIX}/lib/libcudf_identify_stream_usage_mode_testing.so"

echo "STREAM_IDENTIFY_LIB=${STREAM_IDENTIFY_LIB_MODE_CUDF}"

# Run libcudf and libcudf_kafka gtests from libcudf-tests package
rapids-logger "Run gtests"

cd $CONDA_PREFIX/bin/gtests/libcudf/
export GTEST_CUDF_STREAM_MODE="new_cudf_default"
export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/
export LD_PRELOAD=${STREAM_IDENTIFY_LIB_MODE_CUDF}

ctest -E SPAN_TEST -j20 --output-on-failure

# This one test is specifically designed to test using a thrust device vector,
# so we expect and allow it to include default stream usage.
_allowlist_filter="SpanTest.CanConstructFromDeviceContainers"
GTEST_FILTER="-${_allowlist_filter}" ctest -R SPAN_TEST -VV
LD_PRELOAD= GTEST_CUDF_STREAM_MODE=default GTEST_FILTER="${_allowlist_filter}" ctest -R SPAN_TEST -VV

SUITEERROR=$?

if (( ${SUITEERROR} == 0 )); then
    cd $CONDA_PREFIX/bin/gtests/libcudf_kafka/
    ctest -j20 --output-on-failure
    SUITEERROR=$?
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
