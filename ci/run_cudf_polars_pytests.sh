#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

HELPER="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/utils/get_device_and_worker_count.py"
eval "$(python "$HELPER" | sed 's/^/export /')"

# Cap the number of pytest workers (default 16)
: "${MAX_PYTEST_WORKERS:=16}"
if (( NUM_WORKERS > MAX_PYTEST_WORKERS )); then
  NUM_WORKERS="${MAX_PYTEST_WORKERS}"
fi

# Set the GPU to use, if one was selected
if [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi

PYTEST_XDIST_ARGS=(-n "${NUM_WORKERS}" --dist loadfile)

# It is essential to cd into python/cudf_polars as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_cudf_polars_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cudf_polars/

rapids-logger "Running tests with GPU: $GPU_ID and $NUM_WORKERS pytest-xdist workers"

# Test the "in-memory" executor
rapids-logger "Test the in-memory executor"
python -m pytest --cache-clear "$@" "${PYTEST_XDIST_ARGS[@]}" tests --executor in-memory

# Test the default "streaming" executor
rapids-logger "Test the streaming executor"
python -m pytest --cache-clear "$@" "${PYTEST_XDIST_ARGS[@]}" tests --executor streaming

# Test the "streaming" executor with small blocksize
rapids-logger "Test the streaming executor with a small blocksize"
python -m pytest --cache-clear "$@" "${PYTEST_XDIST_ARGS[@]}" tests --executor streaming --blocksize-mode small

# Run experimental tests with Distributed cluster
rapids-logger "Run the experimental tests with the distributed scheduler"
python -m pytest --cache-clear "$@" "tests/experimental" \
    --executor streaming \
    --cluster distributed \
    --cov-fail-under=0  # No code-coverage requirement for these tests.
