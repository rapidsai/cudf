#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Test cudf_polars with rapidsmpf integration
# This script runs experimental tests with single cluster mode and the rapidsmpf runtime

# It is essential to cd into python/cudf_polars as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_cudf_polars_with_rapidsmpf_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cudf_polars/

rapids-logger "Multi-GPU Polars: testing on '--cluster single'"
timeout 10m python -m pytest --cache-clear "$@" "tests/experimental" \
    --executor streaming \
    --runtime rapidsmpf \
    --cluster single

rapids-logger "Multi-GPU Polars: testing on '--cluster distributed'"
timeout 10m python -m pytest --cache-clear "$@" "tests/experimental" \
    --executor streaming \
    --runtime rapidsmpf \
    --cluster distributed

rapids-logger "Multi-GPU Polars: testing on multi-ranks using 'rrun'"
timeout 10m rrun --tag-output -n 2 -g 0,0 python -m pytest --cache-clear "$@" \
    tests/experimental/rapidsmpf/test_spmd.py \
    --executor streaming \
    --runtime=rapidsmpf
