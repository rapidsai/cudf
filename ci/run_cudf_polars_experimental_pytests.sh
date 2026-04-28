#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Test cudf_polars experimental.

# It is essential to cd into python/cudf_polars as `pytest-xdist` + `coverage` seem to work only at this directory level.
# Support invoking outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cudf_polars/

echo "Running the full cudf-polars test suite with both the in-memory and spmd engine"
timeout 10m python -m pytest --cache-clear "$@" tests --ignore=tests/experimental/legacy

echo "Running experimental legacy tests with the 'rapidsmpf' runtime and a 'distributed' cluster"
timeout 10m python -m pytest --cache-clear "$@" "tests/experimental/legacy" \
    --executor streaming \
    --cluster distributed \
    --runtime rapidsmpf
