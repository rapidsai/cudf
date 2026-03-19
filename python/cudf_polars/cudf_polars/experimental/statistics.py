# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracking scan statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import DataFrameScan, Scan
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.base import StatsCollector

if TYPE_CHECKING:
    from cudf_polars.dsl.ir import IR
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


def collect_statistics(
    root: IR,
    config_options: ConfigOptions[StreamingExecutor],
) -> StatsCollector:
    """Collect DataSourceInfo for each leaf Scan/DataFrameScan node."""
    from cudf_polars.experimental.io import (
        _build_dataframe_source,
        _build_parquet_source,
    )

    max_footer = config_options.parquet_options.max_footer_samples
    max_rg = config_options.parquet_options.max_row_group_samples

    # Group parquet Scan nodes by paths, accumulating the union of needed columns
    # across all Scan nodes that read the same files.
    parquet_groups: dict[tuple[str, ...], tuple[set[str], list[Scan]]] = {}
    dataframe_scans: list[DataFrameScan] = []
    for node in traversal([root]):
        if isinstance(node, Scan):
            if node.typ == "parquet":
                paths_key = tuple(node.paths)
                if paths_key not in parquet_groups:
                    parquet_groups[paths_key] = (set(), [])
                parquet_groups[paths_key][0].update(node.schema.keys())
                parquet_groups[paths_key][1].append(node)
        elif isinstance(node, DataFrameScan):
            dataframe_scans.append(node)

    stats = StatsCollector()

    for paths_key, (needed_cols, scan_nodes) in parquet_groups.items():
        source = _build_parquet_source(
            paths_key, frozenset(needed_cols), max_footer, max_rg
        )
        for node in scan_nodes:
            stats.scan_stats[node] = source

    for node in dataframe_scans:
        stats.scan_stats[node] = _build_dataframe_source(node)

    return stats
