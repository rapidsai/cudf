#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support customizing the benchmarks' install location
# First, try the installed location (CI/conda environments)
installed_benchmark_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/benchmarks/libcudf/"
# Fall back to the build directory (devcontainer environments)
devcontainers_benchmark_location="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../cpp/build/latest/benchmarks/"

if [[ -d "${installed_benchmark_location}" ]]; then
    cd "${installed_benchmark_location}"
elif [[ -d "${devcontainers_benchmark_location}" ]]; then
    cd "${devcontainers_benchmark_location}"
else
    echo "Error: Benchmark location not found. Searched:" >&2
    echo "  - ${installed_benchmark_location}" >&2
    echo "  - ${devcontainers_benchmark_location}" >&2
    exit 1
fi

# Ensure that benchmarks are runnable
# Run a small nvbench benchmark
./STRINGS_NVBENCH --profile --benchmark 0 --devices 0
