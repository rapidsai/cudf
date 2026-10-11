#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

integration_timestamp_ms() {
    local timestamp=${EPOCHREALTIME/./}
    printf '%s\n' "$((10#${timestamp} / 1000))"
}

integration_run_timed() {
    local library=$1 phase=$2
    shift 2
    local started_ms status=0 elapsed_ms
    started_ms=$(integration_timestamp_ms)
    "$@" || status=$?
    elapsed_ms=$(($(integration_timestamp_ms) - started_ms))
    printf 'INTEGRATION_TIMING library=%s phase=%s elapsed_ms=%s status=%s\n' \
        "$library" "$phase" "$elapsed_ms" "$status" >&2
    return "$status"
}
