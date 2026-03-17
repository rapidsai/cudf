#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
python -m pytest --cache-clear "$@" "tests/experimental" \
    --executor streaming \
    --cluster distributed \
    --cov-fail-under=0  # No code-coverage requirement for these tests.
