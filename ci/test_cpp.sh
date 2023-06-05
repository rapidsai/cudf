#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

source "$(dirname "$0")/test_cpp_common.sh"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run libcudf and libcudf_kafka gtests from libcudf-tests package
rapids-logger "Run gtests"

cd $CONDA_PREFIX/bin/gtests/libcudf/
export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/

ctest -j20 --output-on-failure

SUITEERROR=$?

if (( ${SUITEERROR} == 0 )); then
    cd $CONDA_PREFIX/bin/gtests/libcudf_kafka/
    ctest -j20 --output-on-failure
    SUITEERROR=$?
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
