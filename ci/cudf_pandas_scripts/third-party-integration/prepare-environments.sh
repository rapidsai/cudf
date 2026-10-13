#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

main() {
    local dependencies_yaml=$1 environment=$2
    shift 2
    local lib channels=() file_keys=()
    for lib in "$@"; do
        file_keys+=(--file-key "test_${lib}")
    done
    if [[ $RAPIDS_BUILD_TYPE == pull-request ]]; then
        channels=(--prepend-channel "$CPP_CHANNEL" --prepend-channel "$PYTHON_CHANNEL" --prepend-channel "$PYTHON_NOARCH_CHANNEL")
    fi
    rapids-dependency-file-generator \
        --config "$dependencies_yaml" --output conda "${file_keys[@]}" \
        --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
        "${channels[@]}" | tee "${environment}.yaml"
    rapids-mamba-retry env create --yes \
        -f "${environment}.yaml" --prefix "$environment"
}

main "$@"
