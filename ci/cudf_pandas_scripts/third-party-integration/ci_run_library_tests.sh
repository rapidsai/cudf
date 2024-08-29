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

    pytest \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --numprocesses=${NUM_PROCESSES} \
    --dist=worksteal \
    ${TEST_DIR}/test_${lib}*.py
}

runtest_cudf_pandas() {
    local lib=$1

    pytest \
    -p cudf.pandas \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --numprocesses=${NUM_PROCESSES} \
    --dist=worksteal \
    ${TEST_DIR}/test_${lib}*.py
}

main() {
    local lib=$1

    # generation phase
    runtest_gold ${lib}
    runtest_cudf_pandas ${lib}

    # assertion phase
    pytest \
    --compare \
    -p cudf.pandas \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --numprocesses=${NUM_PROCESSES} \
    --dist=worksteal \
    ${TEST_DIR}/test_${lib}*.py
}

main $@
