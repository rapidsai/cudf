#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Run cudf_polars experimental tests with --cluster single"

# Run only the experimental tests with --cluster single
# (other cluster modes require multi-node setup)
python -m pytest \
    -v \
    --cache-clear \
    --tb=native \
    --cluster single \
    tests/experimental/
