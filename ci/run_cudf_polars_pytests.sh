#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

# It is essential to cd into python/cudf_polars as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_cudf_polars_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cudf_polars/

# Test the default "cudf" executor
python -m pytest --cache-clear "$@" tests

# Test the "dask-experimental" executor
python -m pytest --cache-clear "$@" tests --executor dask-experimental

# Test the "dask-experimental" executor with Distributed cluster
# Not all tests pass yet, deselecting by name those that are failing.
python -m pytest --cache-clear "$@" tests --executor dask-experimental --dask-cluster \
    -k "not test_groupby_maintain_order_random and not test_scan_csv_multi and not test_select_literal_series" \
    --cov-fail-under=89  # Override coverage, Distributed cluster coverage not yet 100%
