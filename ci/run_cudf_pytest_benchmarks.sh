#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# It is essential to cd into python/cudf as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_cudf_pytest_benchmarks.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cudf/

CUDF_BENCHMARKS_DEBUG_ONLY=ON \
pytest --cache-clear "$@" benchmarks
