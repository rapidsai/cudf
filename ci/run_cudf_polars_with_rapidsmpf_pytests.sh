#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Test cudf_polars with rapidsmpf integration
# This script runs experimental tests with single cluster mode and the rapidsmpf runtime

# It is essential to cd into python/cudf_polars as `pytest-xdist` + `coverage` seem to work only at this directory level.

# Support invoking run_cudf_polars_with_rapidsmpf_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cudf_polars/

# TODO: why do we need this?
ucxx_lib64="$(python -c 'import ucxx._lib.libucxx as m, pathlib; print(pathlib.Path(m.__file__).resolve().parent.parent / "lib64")')"
if [ -d "$ucxx_lib64" ]; then
    export LD_LIBRARY_PATH="$ucxx_lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Run experimental tests with the "single" cluster mode and the "rapidsmpf" runtime
rapids-logger "Running experimental tests with the 'rapidsmpf' runtime and a 'single' cluster"
timeout 10m python -m pytest --cache-clear "$@" "tests/experimental" \
    --executor streaming \
    --cluster single \
    --runtime rapidsmpf

# Run experimental tests with the "distributed" cluster mode and the "rapidsmpf" runtime
rapids-logger "Running experimental tests with the 'rapidsmpf' runtime and a 'distributed' cluster"
timeout 10m python -m pytest --cache-clear "$@" "tests/experimental" \
    --executor streaming \
    --cluster distributed \
    --runtime rapidsmpf
