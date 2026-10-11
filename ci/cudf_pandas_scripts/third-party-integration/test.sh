#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# shellcheck source=ci/cudf_pandas_scripts/third-party-integration/timing.sh
source "$(dirname "${BASH_SOURCE[0]}")/timing.sh"

prepare_environment() {
    local lib=$1 directory="${INTEGRATION_RUN_DIR}/$1"
    local channels=()
    if [[ $RAPIDS_BUILD_TYPE == pull-request ]]; then
        channels=(--prepend-channel "$CPP_CHANNEL" --prepend-channel "$PYTHON_CHANNEL" --prepend-channel "$PYTHON_NOARCH_CHANNEL")
    fi
    integration_run_timed "$lib" dependencies rapids-dependency-file-generator \
        --config "$INTEGRATION_DEPENDENCIES_YAML" --output conda --file-key "test_${lib}" \
        --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
        "${channels[@]}" | tee "${directory}/env.yaml" || return 1
    integration_run_timed "$lib" environment rapids-mamba-retry env create --yes \
        -f "${directory}/env.yaml" --prefix "${directory}/env"
}

run_library() {
    local lib=$1
    # Each test shell owns its activation; preparing another environment must not change it.
    set +u
    source /opt/conda/etc/profile.d/conda.sh
    integration_run_timed "$lib" activation conda activate "${INTEGRATION_RUN_DIR}/${lib}/env"
    set -u

    integration_run_timed "$lib" environment-diagnostics rapids-print-env
    integration_run_timed "$lib" package-manifest conda list --explicit
    integration_run_timed "$lib" gpu-diagnostics nvidia-smi

    local workers=${NUM_PROCESSES:-8}
    local comparison_workers=${COMPARE_NUM_PROCESSES:-$workers}
    if [[ $lib == tensorflow ]]; then
        workers=1
        comparison_workers=1
    fi
    TEST_DIR="${INTEGRATION_REPO_ROOT}/python/cudf/cudf_pandas_tests/third_party_integration_tests/tests" \
    NUM_PROCESSES=$workers \
    COMPARE_NUM_PROCESSES=$comparison_workers \
        integration_run_timed "$lib" tests \
        bash "${INTEGRATION_REPO_ROOT}/ci/cudf_pandas_scripts/third-party-integration/run-library-tests.sh" "$lib"
}

prepare_pid=0
test_pid=0
INTEGRATION_RUN_DIR=${INTEGRATION_RUN_DIR:-}
script_started_ms=0

cleanup() {
    local pid
    for pid in "$prepare_pid" "$test_pid"; do
        if ((pid > 0)); then
            # Terminate the entire isolated process group, including solvers and pytest workers.
            kill -TERM -- "-${pid}" 2>/dev/null || true
            kill -KILL -- "-${pid}" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    if [[ -n $INTEGRATION_RUN_DIR ]]; then
        rm -rf -- "$INTEGRATION_RUN_DIR"
    fi
}

finish() {
    local status=$?
    cleanup || status=1
    printf 'INTEGRATION_TIMING library=all phase=script elapsed_ms=%s status=%s\n' \
        "$(($(integration_timestamp_ms) - script_started_ms))" "$status" >&2
    return "$status"
}

launch_preparation() {
    local lib=$1
    mkdir -p "${INTEGRATION_RUN_DIR}/${lib}"
    setsid bash "${BASH_SOURCE[0]}" --prepare "$lib" \
        > "${INTEGRATION_RUN_DIR}/${lib}/prepare.log" 2>&1 &
    prepare_pid=$!
}

main() {
    script_started_ms=$(integration_timestamp_ms)
    local dependencies_yaml=$1 prefetch=${INTEGRATION_ENV_PREFETCH:-0}
    if [[ $prefetch != 0 && $prefetch != 1 ]]; then
        printf 'INTEGRATION_ENV_PREFETCH must be 0 or 1.\n' >&2
        return 1
    fi
    local library_json
    library_json=$(yq -o json "$dependencies_yaml" | jq -c '.files | keys | map(select(startswith("test_")) | sub("^test_"; ""))')
    local libraries=()
    mapfile -t libraries < <(jq -r '.[]' <<< "$library_json")
    if ((${#libraries[@]} == 0)); then
        printf 'No integration libraries selected.\n' >&2
        return 1
    fi
    local lib
    for lib in "${libraries[@]}"; do
        if [[ ! $lib =~ ^[a-zA-Z0-9_-]+$ ]]; then
            printf 'Invalid library identifier: %s\n' "$lib" >&2
            return 1
        fi
    done
    integration_run_timed all package-manager-version mamba --version
    integration_run_timed all available-memory free -m
    integration_run_timed all available-disk df -h . /tmp
    if [[ $RAPIDS_BUILD_TYPE == pull-request ]]; then
        CPP_CHANNEL=$(integration_run_timed all download-cpp rapids-download-from-github "$(rapids-artifact-name conda_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
        PYTHON_CHANNEL=$(integration_run_timed all download-python rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
        PYTHON_NOARCH_CHANNEL=$(integration_run_timed all download-noarch rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --pure --cuda "$RAPIDS_CUDA_VERSION" --arch any)")
        export CPP_CHANNEL PYTHON_CHANNEL PYTHON_NOARCH_CHANNEL
    fi
    INTEGRATION_REPO_ROOT=$(git rev-parse --show-toplevel)
    INTEGRATION_RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/cudf-pandas-integration.XXXXXX")
    INTEGRATION_DEPENDENCIES_YAML=$dependencies_yaml
    export INTEGRATION_REPO_ROOT INTEGRATION_RUN_DIR INTEGRATION_DEPENDENCIES_YAML
    trap finish EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    local index status=0 ready next_started test_status
    launch_preparation "${libraries[0]}"
    for ((index = 0; index < ${#libraries[@]}; index++)); do
        lib=${libraries[index]}
        ready=0
        if wait "$prepare_pid"; then
            ready=1
        else
            status=1
        fi
        prepare_pid=0
        printf '::group::Prepare integration environment: %s\n' "$lib"
        cat "${INTEGRATION_RUN_DIR}/${lib}/prepare.log"
        printf '::endgroup::\n'

        next_started=0
        if [[ $prefetch == 1 ]] && ((index + 1 < ${#libraries[@]})); then
            launch_preparation "${libraries[index + 1]}"
            next_started=1
        fi
        if ((ready)); then
            printf '::group::Integration tests: %s\n' "$lib"
            setsid timeout 45m bash "${BASH_SOURCE[0]}" --test "$lib" &
            test_pid=$!
            if wait "$test_pid"; then
                test_status=0
            else
                test_status=$?
                status=1
            fi
            test_pid=0
            printf 'INTEGRATION_LIBRARY library=%s status=%s\n' "$lib" "$test_status"
            printf '::endgroup::\n'
        fi
        if [[ -f "${INTEGRATION_RUN_DIR}/${lib}/env/conda-meta/history" ]]; then
            printf '::group::Remove integration environment: %s\n' "$lib"
            if ! integration_run_timed "$lib" cleanup rapids-mamba-retry env remove --yes \
                --prefix "${INTEGRATION_RUN_DIR}/${lib}/env"; then
                printf '::endgroup::\n'
                printf 'Failed to remove %s environment; stopping before creating more environments.\n' "$lib" >&2
                return 1
            fi
            printf '::endgroup::\n'
        fi
        if ((next_started == 0 && index + 1 < ${#libraries[@]})); then
            launch_preparation "${libraries[index + 1]}"
        fi
    done
    return "$status"
}

case ${1:-} in
    --prepare) prepare_environment "$2" ;;
    --test) run_library "$2" ;;
    *) main "$@" ;;
esac
