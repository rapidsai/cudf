#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/benchmarks/libcudf/";

# Run all nvbench benchmarks with --profile and 1 minute timeout
for bench in *_NVBENCH; do
  if [[ -x "$bench" && -f "$bench" ]]; then
    echo "Running $bench with --profile..."
    timeout 8m "./$bench" --profile --devices 0 -q --rmm_mode cuda
  fi
done
