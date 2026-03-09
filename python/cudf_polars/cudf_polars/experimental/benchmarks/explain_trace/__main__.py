# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Entry point for running explain_trace as a module."""

from __future__ import annotations

from cudf_polars.experimental.benchmarks.explain_trace._core import main

if __name__ == "__main__":
    main()
