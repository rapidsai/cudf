#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking test_cpp.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

source ./ci/test_cpp_common.sh

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run libcudf and libcudf_kafka gtests from libcudf-tests package
export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/

rapids-logger "Run libcudf gtests"
./ci/run_cudf_ctests.sh -j20
SUITEERROR=$?

if (( SUITEERROR == 0 )); then
    rapids-logger "Run libcudf examples"
    ./ci/run_cudf_examples.sh
    SUITEERROR=$?
fi

if (( SUITEERROR == 0 )); then
    rapids-logger "Run libcudf_kafka gtests"
    ./ci/run_cudf_kafka_ctests.sh -j20
    SUITEERROR=$?
fi

# Ensure that benchmarks are runnable
rapids-logger "Run tests of libcudf benchmarks"

if (( SUITEERROR == 0 )); then
    ./ci/run_cudf_benchmark_smoketests.sh
    SUITEERROR=$?
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
