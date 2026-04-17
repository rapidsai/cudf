#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# It is essential to cd into python/cudf_polars as `pytest-xdist` + `coverage` seem to work only at this directory level.
# Support invoking run_cudf_polars_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cudf_polars/

# Run all non-experimental tests using both the in-memory and streaming executor.
IGNORE_EXPERIMENTAL="--ignore=tests/experimental/"
python -m pytest --cache-clear "$@" tests $IGNORE_EXPERIMENTAL --executor in-memory
python -m pytest --cache-clear "$@" tests $IGNORE_EXPERIMENTAL --executor streaming
python -m pytest --cache-clear "$@" tests $IGNORE_EXPERIMENTAL --executor streaming \
    --blocksize-mode small
# Run all tests with --runtime rapidsmpf
python -m pytest --cache-clear "$@" tests --executor streaming \
    --runtime rapidsmpf \
    --blocksize-mode small
