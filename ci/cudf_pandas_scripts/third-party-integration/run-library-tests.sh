#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cleanup() {
    rm "${TEST_DIR}"/results-*.pickle
}

trap cleanup EXIT

runtest() {
    local lib=$1
    local mode=$2

    local plugin=""
    if [ "$mode" = "cudf" ]; then
        plugin="-p cudf.pandas"
    fi

    echo "Running tests for $lib in $mode mode"
    pytest \
    "$plugin" \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --numprocesses="${NUM_PROCESSES}" \
    --dist=worksteal \
    "${TEST_DIR}"/test_"${lib}"*.py
}

main() {
    local lib=$1

    # generation phase
    runtest "${lib}" "gold"
    echo "finding"
    find "/__w/" -type f -name "*.pickle"
    runtest "${lib}" "cudf"
    find "/__w/" -type f -name "*.pickle"

    # assertion phase
    pytest \
    --compare \
    -p cudf.pandas \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --numprocesses="${NUM_PROCESSES}" \
    --dist=worksteal \
    "${TEST_DIR}"/test_"${lib}"*.py
}

main "$@"
