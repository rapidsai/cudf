# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""I/O utilities for cudf-polars."""

from __future__ import annotations

from cudf_polars.io.parquet_hive import expand_hive_scan

__all__: list[str] = ["expand_hive_scan"]
