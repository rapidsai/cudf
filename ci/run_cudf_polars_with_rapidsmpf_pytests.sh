#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Test cudf_polars with rapidsmpf integration
# This script runs experimental tests with single cluster mode and the rapidsmpf runtime

# It is essential to cd into python/cudf_polars as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_cudf_polars_with_rapidsmpf_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cudf_polars/

# Run experimental tests with the "single" cluster mode and the "rapidsmpf" runtime
rapids-logger "Running experimental tests with the 'rapidsmpf' runtime"
python -m pytest --cache-clear "$@" "tests/experimental" \
    --executor streaming \
    --cluster single \
    --runtime rapidsmpf
