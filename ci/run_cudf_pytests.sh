#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# It is essential to cd into python/cudf/cudf as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_cudf_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cudf/cudf/

pytest --cache-clear --ignore="benchmarks" "$@" tests
