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
    "${TEST_DIR}"/test_"${lib}"*.py
    # --numprocesses="${NUM_PROCESSES}" \
    # --dist=worksteal \

}

main() {
    local lib=$1

    # generation phase
    runtest "${lib}" "gold"
    runtest "${lib}" "cudf"

    # assertion phase
    pytest \
    --compare \
    -p cudf.pandas \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    "${TEST_DIR}"/test_"${lib}"*.py
    # --numprocesses="${NUM_PROCESSES}" \
    # --dist=worksteal \

}

main "$@"
