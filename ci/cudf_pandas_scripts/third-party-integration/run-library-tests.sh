#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# shellcheck source=ci/cudf_pandas_scripts/third-party-integration/timing.sh
source "$(dirname "${BASH_SOURCE[0]}")/timing.sh"

set -euo pipefail

cleanup() {
    rm -f "${TEST_DIR}"/results-*.pickle
}

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
    local lib=$1 environment=$2 repo_root
    repo_root=$(git rev-parse --show-toplevel)
    TEST_DIR=${TEST_DIR:-${repo_root}/python/cudf/cudf_pandas_tests/third_party_integration_tests/tests}
    trap cleanup EXIT

    # Activation stays in the test process, independent of background preparation.
    set +u
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "$environment"
    set -u

    rapids-print-env
    conda list --explicit
    nvidia-smi

    NUM_PROCESSES=${NUM_PROCESSES:-0}
    COMPARE_NUM_PROCESSES=${COMPARE_NUM_PROCESSES:-0}
    if [[ $lib == tensorflow ]]; then
        NUM_PROCESSES=1
        COMPARE_NUM_PROCESSES=1
    fi
    export TEST_DIR NUM_PROCESSES COMPARE_NUM_PROCESSES
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
