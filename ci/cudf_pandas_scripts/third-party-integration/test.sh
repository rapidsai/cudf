#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

prepare_pid=0
INTEGRATION_RUN_DIR=${INTEGRATION_RUN_DIR:-}

cleanup() {
    if ((prepare_pid > 0)); then
        # Retry-helper timeouts create additional process groups inside this session.
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
    local dependencies_yaml=$1 scripts_dir
    scripts_dir=$(dirname "${BASH_SOURCE[0]}")
    local library_json
    library_json=$(yq -o json "$dependencies_yaml" | jq -c '.files | keys | map(select(startswith("test_")) | sub("^test_"; ""))')
    local libraries=()
    mapfile -t libraries < <(jq -r '.[]' <<< "$library_json")
    if ((${#libraries[@]} == 0)); then
        printf 'No integration libraries selected.\n' >&2
        return 1
    fi
    if [[ $RAPIDS_BUILD_TYPE == pull-request ]]; then
        CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcudf cudf --cuda "$RAPIDS_CUDA_VERSION")")
        PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --stable --cuda "$RAPIDS_CUDA_VERSION")")
        PYTHON_NOARCH_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cudf cudf --pure --cuda "$RAPIDS_CUDA_VERSION" --arch any)")
        export CPP_CHANNEL PYTHON_CHANNEL PYTHON_NOARCH_CHANNEL
    fi
    INTEGRATION_RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/cudf-pandas-integration.XXXXXX")
    INTEGRATION_DEPENDENCIES_YAML=$dependencies_yaml
    export INTEGRATION_RUN_DIR INTEGRATION_DEPENDENCIES_YAML
    trap finish EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    local status=0 test_status lib
    bash "${scripts_dir}/prepare-environments.sh" "${libraries[0]}"
    if ((${#libraries[@]} > 1)); then
        # A background function would share the coordinator's session. Keep the producer
        # in a separate script so setsid isolates all its descendants for session-based
        # cleanup, including the additional process groups created by nested timeouts.
        setsid bash "${scripts_dir}/prepare-environments.sh" "${libraries[@]:1}" &
        prepare_pid=$!
    fi
    for lib in "${libraries[@]}"; do
        while [[ ! -f "${INTEGRATION_RUN_DIR}/${lib}/status" ]]; do
            if ! kill -0 "$prepare_pid" 2>/dev/null; then
                printf 'Environment producer exited before preparing %s.\n' "$lib" >&2
                return 1
            fi
            sleep 1
        done
        printf 'Prepare integration environment: %s\n' "$lib"
        cat "${INTEGRATION_RUN_DIR}/${lib}/prepare.log"

        if [[ $(< "${INTEGRATION_RUN_DIR}/${lib}/status") == 0 ]]; then
            printf 'Integration tests: %s\n' "$lib"
            if timeout 45m bash "${scripts_dir}/run-library-tests.sh" "$lib" "${INTEGRATION_RUN_DIR}/${lib}/env"; then
                test_status=0
            else
                test_status=$?
                status=1
            fi
            printf 'INTEGRATION_LIBRARY library=%s status=%s\n' "$lib" "$test_status"
            printf 'Remove integration environment: %s\n' "$lib"
            if ! rapids-mamba-retry env remove --yes \
                --prefix "${INTEGRATION_RUN_DIR}/${lib}/env"; then
                printf 'Failed to remove %s environment.\n' "$lib" >&2
                return 1
            fi
        else
            printf 'Failed to prepare %s environment.\n' "$lib" >&2
            status=1
        fi
    done
    if ((prepare_pid > 0)); then
        wait "$prepare_pid" || status=1
        prepare_pid=0
    fi
    return "$status"
}

main "$@"
