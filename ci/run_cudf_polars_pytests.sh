#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# It is essential to cd into python/cudf_polars as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_cudf_polars_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cudf_polars/

# Test the default "cudf" executor
python -m pytest --cache-clear "$@" tests

# Test the "dask-experimental" executor
python -m pytest --cache-clear "$@" tests --executor dask-experimental
