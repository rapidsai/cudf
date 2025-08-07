#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/benchmarks/libcudf/";

# Ensure that benchmarks are runnable
# Run a small Google benchmark
./MERGE_BENCH --benchmark_filter=/2/
# Run a small nvbench benchmark
./STRINGS_NVBENCH --profile --benchmark 0 --devices 0
