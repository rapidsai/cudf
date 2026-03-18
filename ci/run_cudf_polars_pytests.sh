#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# It is essential to cd into python/cudf_polars as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_cudf_polars_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cudf_polars/

echo "Test the in-memory executor"
python -m pytest --cache-clear "$@" tests --executor in-memory

echo "Test the default streaming executor"
python -m pytest --cache-clear "$@" tests --executor streaming

echo "Test the streaming executor with small blocksize"
python -m pytest --cache-clear "$@" tests --executor streaming --blocksize-mode small

echo "Test the (future default) streaming executor with rapidsmpf"
CUDF_POLARS__PARQUET_OPTIONS__USE_RAPIDSMPF_NATIVE=1 CUDF_POLARS__EXECUTOR__SHUFFLE_METHOD=rapidsmpf python -m pytest --cache-clear "$@" tests \
    --executor streaming \
    --blocksize-mode small \
    --cluster single \
    --runtime rapidsmpf \
    --cov-fail-under=0  # TODO? missing coverage may be due to old task-based paths

echo "Run experimental tests with Distributed cluster"
python -m pytest --cache-clear "$@" "tests/experimental" \
    --executor streaming \
    --cluster distributed \
    --cov-fail-under=0  # No code-coverage requirement for these tests.

echo "Run experimental tests with the distributed cluster mode and the rapidsmpf runtime"
python -m pytest --cache-clear "$@" "tests/experimental" \
    --executor streaming \
    --cluster distributed \
    --runtime rapidsmpf \
    --cov-fail-under=0  # No code-coverage requirement for these tests(?)
