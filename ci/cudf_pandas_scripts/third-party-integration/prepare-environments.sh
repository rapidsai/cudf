#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

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
            rapids-mamba-retry env create --yes \
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

for library in "$@"; do
    prepare_environment "$library"
done
