#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# shellcheck source=ci/cudf_pandas_scripts/third-party-integration/timing.sh
source "$(dirname "${BASH_SOURCE[0]}")/timing.sh"

prepare_environment() {
    local lib=$1 directory="${INTEGRATION_RUN_DIR}/$1"
    local status=0 channels=()
    mkdir -p "$directory"
    if [[ $RAPIDS_BUILD_TYPE == pull-request ]]; then
        channels=(--prepend-channel "$CPP_CHANNEL" --prepend-channel "$PYTHON_CHANNEL" --prepend-channel "$PYTHON_NOARCH_CHANNEL")
    fi
    {
        if rapids-dependency-file-generator \
            --config "$INTEGRATION_DEPENDENCIES_YAML" --output conda --file-key "test_${lib}" \
            --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
            "${channels[@]}" | tee "${directory}/env.yaml" &&
            integration_run_timed "$lib" environment rapids-mamba-retry env create --yes \
                -f "${directory}/env.yaml" --prefix "${directory}/env"; then
            status=0
        else
            status=$?
        fi
    } > "${directory}/prepare.log" 2>&1
    # Atomic publication prevents consumers from reading an empty status file.
    printf '%s\n' "$status" > "${directory}/status.tmp"
    mv "${directory}/status.tmp" "${directory}/status"
}

run_library() {
    local lib=$1
    # Each test shell owns its activation; preparing another environment must not change it.
    set +u
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "${INTEGRATION_RUN_DIR}/${lib}/env"
    set -u

    rapids-print-env
    conda list --explicit
    nvidia-smi

    local workers=${NUM_PROCESSES:-0}
    local comparison_workers=${COMPARE_NUM_PROCESSES:-0}
    if [[ $lib == tensorflow ]]; then
        workers=1
        comparison_workers=1
    fi
    TEST_DIR="${INTEGRATION_REPO_ROOT}/python/cudf/cudf_pandas_tests/third_party_integration_tests/tests" \
    NUM_PROCESSES=$workers \
    COMPARE_NUM_PROCESSES=$comparison_workers \
        bash "${INTEGRATION_REPO_ROOT}/ci/cudf_pandas_scripts/third-party-integration/run-library-tests.sh" "$lib"
}

prepare_pid=0
INTEGRATION_RUN_DIR=${INTEGRATION_RUN_DIR:-}

cleanup() {
    if ((prepare_pid > 0)); then
        # Retry-helper timeouts create additional process groups inside this session.
        pkill -TERM --session "$prepare_pid" 2>/dev/null || true
        pkill -KILL --session "$prepare_pid" 2>/dev/null || true
        wait "$prepare_pid" 2>/dev/null || true
    fi
    if [[ -n $INTEGRATION_RUN_DIR ]]; then
        rm -rf -- "$INTEGRATION_RUN_DIR"
    fi
}

finish() {
    local status=$?
    cleanup || status=1
    return "$status"
}

main() {
    local dependencies_yaml=$1 prefetch=${INTEGRATION_ENV_PREFETCH:-1}
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
    mamba --version
    free -m
    df -h . /tmp
    if [[ $RAPIDS_BUILD_TYPE == pull-request ]]; then
        CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
        PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
        PYTHON_NOARCH_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --pure --cuda "$RAPIDS_CUDA_VERSION" --arch any)")
        export CPP_CHANNEL PYTHON_CHANNEL PYTHON_NOARCH_CHANNEL
    fi
    INTEGRATION_REPO_ROOT=$(git rev-parse --show-toplevel)
    INTEGRATION_RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/cudf-pandas-integration.XXXXXX")
    INTEGRATION_DEPENDENCIES_YAML=$dependencies_yaml
    export INTEGRATION_REPO_ROOT INTEGRATION_RUN_DIR INTEGRATION_DEPENDENCIES_YAML
    trap finish EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    local status=0 test_status
    if [[ $prefetch == 1 ]]; then
        prepare_environment "${libraries[0]}"
        if ((${#libraries[@]} > 1)); then
            setsid bash "${BASH_SOURCE[0]}" --prepare "${libraries[@]:1}" &
            prepare_pid=$!
        fi
    fi
    for lib in "${libraries[@]}"; do
        if [[ $prefetch == 0 ]]; then
            prepare_environment "$lib"
        fi
        while [[ ! -f "${INTEGRATION_RUN_DIR}/${lib}/status" ]]; do
            if ! kill -0 "$prepare_pid" 2>/dev/null; then
                printf 'Environment producer exited before preparing %s.\n' "$lib" >&2
                return 1
            fi
            sleep 1
        done
        printf '::group::Prepare integration environment: %s\n' "$lib"
        cat "${INTEGRATION_RUN_DIR}/${lib}/prepare.log"
        printf '::endgroup::\n'

        if [[ $(< "${INTEGRATION_RUN_DIR}/${lib}/status") == 0 ]]; then
            printf '::group::Integration tests: %s\n' "$lib"
            if timeout 45m bash "${BASH_SOURCE[0]}" --test "$lib"; then
                test_status=0
            else
                test_status=$?
                status=1
            fi
            printf 'INTEGRATION_LIBRARY library=%s status=%s\n' "$lib" "$test_status"
            printf '::endgroup::\n'
        else
            printf 'Failed to prepare %s environment.\n' "$lib" >&2
            status=1
        fi
        if [[ -f "${INTEGRATION_RUN_DIR}/${lib}/env/conda-meta/history" ]]; then
            printf '::group::Remove integration environment: %s\n' "$lib"
            if ! rapids-mamba-retry env remove --yes \
                --prefix "${INTEGRATION_RUN_DIR}/${lib}/env"; then
                printf '::endgroup::\n'
                printf 'Failed to remove %s environment.\n' "$lib" >&2
                return 1
            fi
            printf '::endgroup::\n'
        fi
    done
    if ((prepare_pid > 0)); then
        wait "$prepare_pid" || status=1
        prepare_pid=0
    fi
    return "$status"
}

case ${1:-} in
    --prepare)
        shift
        for library in "$@"; do
            prepare_environment "$library"
        done
        ;;
    --test) run_library "$2" ;;
    *) main "$@" ;;
esac
