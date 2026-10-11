#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# shellcheck source=ci/cudf_pandas_scripts/third-party-integration/timing.sh
source "$(dirname "${BASH_SOURCE[0]}")/timing.sh"

set -uo pipefail

cleanup() {
    rm -f "${TEST_DIR}"/results-*.pickle
}

trap cleanup EXIT

runtest() {
    local lib=$1
    local mode=$2

    echo "Running tests for $lib in $mode mode"
    local plugin=()
    if [ "$mode" = "cudf" ]; then
        plugin=("-p cudf.pandas")
    fi

    integration_run_timed "$lib" "$mode" pytest \
    "${plugin[@]}" \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --durations=50 --durations-min=1 \
    --numprocesses="${NUM_PROCESSES}" \
    --dist=worksteal \
    "${TEST_DIR}"/test_"${lib}"*.py
}

main() {
    local lib=$1
    local status=0 workers
    for workers in "${NUM_PROCESSES}" "${COMPARE_NUM_PROCESSES:-${NUM_PROCESSES}}"; do
        if [[ ! $workers =~ ^(0|[1-9][0-9]*)$ ]]; then
            printf 'Worker counts must be nonnegative integers: %s\n' "$workers" >&2
            return 1
        fi
    done

    # generation phase
    runtest "${lib}" "gold" || status=1
    runtest "${lib}" "cudf" || status=1

    # assertion phase
    integration_run_timed "$lib" compare pytest \
    --compare \
    -p cudf.pandas \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --durations=50 --durations-min=1 \
    --numprocesses="${COMPARE_NUM_PROCESSES:-${NUM_PROCESSES}}" \
    --dist=worksteal \
    "${TEST_DIR}"/test_"${lib}"*.py || status=1

    return "$status"
}

main "$@"
