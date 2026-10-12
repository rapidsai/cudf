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
    local workers=${NUM_PROCESSES} args=()
    case $mode in
        cudf) args=("-p cudf.pandas") ;;
        compare)
            args=(--compare -p cudf.pandas)
            workers=${COMPARE_NUM_PROCESSES:-${NUM_PROCESSES}}
            ;;
    esac

    integration_run_timed "$lib" "$mode" pytest \
    "${args[@]}" \
    -v \
    --continue-on-collection-errors \
    --cache-clear \
    --durations=50 --durations-min=1 \
    --numprocesses="${workers}" \
    --dist=worksteal \
    "${TEST_DIR}"/test_"${lib}"*.py
}

main() {
    local lib=$1
    local status=0 workers mode
    for workers in "${NUM_PROCESSES}" "${COMPARE_NUM_PROCESSES:-${NUM_PROCESSES}}"; do
        if [[ ! $workers =~ ^(0|[1-9][0-9]*)$ ]]; then
            printf 'Worker counts must be nonnegative integers: %s\n' "$workers" >&2
            return 1
        fi
    done

    for mode in gold cudf compare; do
        runtest "${lib}" "$mode" || status=1
    done

    return "$status"
}

main "$@"
