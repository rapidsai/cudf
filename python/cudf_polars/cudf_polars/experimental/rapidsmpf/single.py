# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Single-worker execution with the streaming RapidsMPF runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rapidsmpf.integrations.single import get_worker_context
from rapidsmpf.streaming.core.context import Context


if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata

    import polars as pl

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo, StatsCollector
    from cudf_polars.experimental.parallel import ConfigOptions


def evaluate_pipeline_single(
    callback,
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
    stats: StatsCollector,
    collective_id_map: dict[IR, list[int]],
    *,
    collect_metadata: bool = False,
) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Evaluate a RapidsMPF streaming pipeline using the single-worker context.

    Uses the worker context from :func:`rapidsmpf.integrations.single.setup_worker`
    (must be called beforehand, e.g. via :func:`initialize_single_or_dask_cluster`).

    Parameters
    ----------
    callback
        The pipeline evaluation callback (e.g. :func:`evaluate_pipeline`).
    ir
        The IR node.
    partition_info
        Partition information.
    config_options
        Configuration options.
    stats
        Statistics collector.
    collective_id_map
        Mapping from Shuffle/Repartition/Join IR nodes to reserved collective IDs.
    collect_metadata
        Whether to collect metadata.

    Returns
    -------
    The output DataFrame and metadata collector.
    """
    worker_ctx = get_worker_context()
    assert worker_ctx.comm is not None

    with Context(
        worker_ctx.comm.logger, worker_ctx.br, worker_ctx.options
    ) as rmpf_context:
        return callback(
            ir,
            partition_info,
            config_options,
            stats,
            collective_id_map,
            worker_ctx.comm,
            rmpf_context=rmpf_context,
            collect_metadata=collect_metadata,
        )
