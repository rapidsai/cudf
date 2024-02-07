#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

if [ -d "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcudf/" ]; then
    # Support customizing the ctests' install location
    cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcudf/";
    ctest --output-on-failure "$@"
fi

# Ensure that benchmarks are runnable
if [ -d "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/benchmarks/libcudf/" ]; then
    # Support customizing the ctests' install location
    cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/benchmarks/libcudf/";
    # Run a small Google benchmark
    ./MERGE_BENCH --benchmark_filter=/2/
    # Run a small nvbench benchmark
    ./STRINGS_NVBENCH --run-once --benchmark 0 --devices 0
fi
