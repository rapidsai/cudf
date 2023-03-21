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
        GTEST_CUDF_STREAM_MODE="new_cudf_default" LD_PRELOAD=${STREAM_IDENTIFY_LIB_MODE_CUDF} ${gt} --gtest_output=xml:${RAPIDS_TESTS_DIR} --gtest_filter="-${gtest_filter}" && \
            ${gt} --gtest_output=xml:${RAPIDS_TESTS_DIR} --gtest_filter="${gtest_filter}"
    else
        GTEST_CUDF_STREAM_MODE="new_cudf_default" LD_PRELOAD=${STREAM_IDENTIFY_LIB_MODE_CUDF} ${gt} --gtest_output=xml:${RAPIDS_TESTS_DIR}
    fi
done

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
