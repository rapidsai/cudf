# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for tracking scan statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.ir import DataFrameScan, Scan
from cudf_polars.dsl.traversal import post_traversal
from cudf_polars.experimental.base import StatsCollector

if TYPE_CHECKING:
    from cudf_polars.dsl.ir import IR
    from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


def collect_statistics(
    root: IR,
    config_options: ConfigOptions[StreamingExecutor],
) -> StatsCollector:
    """
    Collect scan statistics for a query.

    Parameters
    ----------
    root
        Root IR node.
    config_options
        GPUEngine configuration options.

    Returns
    -------
    A StatsCollector with scan_stats populated for leaf Scan/DataFrameScan nodes.
    """
    if config_options.executor.stats_planning.use_io_partitioning:
        return collect_base_stats(root, config_options)
    return StatsCollector()


def collect_base_stats(
    root: IR,
    config_options: ConfigOptions[StreamingExecutor],
) -> StatsCollector:
    """Collect DataSourceInfo for each leaf Scan/DataFrameScan node."""
    from cudf_polars.experimental.io import _make_datasource_info

    stats = StatsCollector()
    for node in post_traversal([root]):
        if isinstance(node, (Scan, DataFrameScan)):
            source = _make_datasource_info(node, config_options)
            if source is not None:
                stats.scan_stats[node] = source
    return stats
