#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

# It is essential to cd into python/cudf_polars as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_cudf_polars_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cudf_polars/

# Test the "in-memory" executor
python -m pytest --cache-clear "$@" tests --executor in-memory

# Test the default "streaming" executor
python -m pytest --cache-clear "$@" tests --executor streaming

# Test the "streaming" executor with small blocksize
python -m pytest --cache-clear "$@" tests --executor streaming --blocksize-mode small

# Run experimental tests with Distributed cluster
# Runtime in CI tends to increase when running these tests with multiple processes.
# https://github.com/rapidsai/cudf/pull/19980#issuecomment-3340039980
python -m pytest --cache-clear "$@" "tests/experimental" \
    --executor streaming \
    --scheduler distributed \
    --numprocesses=0 \
    --cov-fail-under=0  # No code-coverage requirement for these tests.
