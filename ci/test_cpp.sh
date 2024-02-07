#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

source "$(dirname "$0")/test_cpp_common.sh"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run libcudf and libcudf_kafka gtests from libcudf-tests package
export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/

pushd $CONDA_PREFIX/bin/gtests/libcudf/
rapids-logger "Run libcudf gtests"
ctest -j20 --output-on-failure --no-tests=error
SUITEERROR=$?
popd

if (( ${SUITEERROR} == 0 )); then
    pushd $CONDA_PREFIX/bin/gtests/libcudf_kafka/
    rapids-logger "Run libcudf_kafka gtests"
    ctest -j20 --output-on-failure --no-tests=error
    SUITEERROR=$?
    popd
fi

# Ensure that benchmarks are runnable
pushd $CONDA_PREFIX/bin/benchmarks/libcudf/
rapids-logger "Run tests of libcudf benchmarks"

if (( ${SUITEERROR} == 0 )); then
    # Run a small Google benchmark
    ./MERGE_BENCH --benchmark_filter=/2/
    SUITEERROR=$?
fi

if (( ${SUITEERROR} == 0 )); then
    # Run a small nvbench benchmark
    ./STRINGS_NVBENCH --run-once --benchmark 0 --devices 0
    SUITEERROR=$?
fi
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
