#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cleanup() {
    rm ${TEST_DIR}/results-*.pickle
}

trap cleanup EXIT

runtest_gold() {
    local lib=$1
    local test_keys=${@:2}

    pytest \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-${lib}-gold.xml" \
    --numprocesses=${NUM_PROCESSES} \
    --dist=worksteal \
    ${TEST_DIR}/test_${lib}*.py \
    ${test_keys}
}

runtest_cudf_pandas() {
    local lib=$1
    local test_keys=${@:2}

    pytest \
    -p cudf.pandas \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-${lib}-cudf-pandas.xml" \
    --numprocesses=${NUM_PROCESSES} \
    --dist=worksteal \
    ${TEST_DIR}/test_${lib}*.py \
    ${test_keys}
}

main() {
    local lib=$1
    local test_keys=${@:2}

    # generation phase
    runtest_gold ${lib} ${test_keys}
    runtest_cudf_pandas ${lib} ${test_keys}

    # assertion phase
    pytest \
    --compare \
    -p cudf.pandas \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-${lib}-assertion.xml" \
    --numprocesses=${NUM_PROCESSES} \
    --dist=worksteal \
    ${TEST_DIR}/test_${lib}*.py \
    ${test_keys}
}

main $@
