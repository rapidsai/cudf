# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Dask-based execution with the streaming RapidsMPF runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from distributed import get_client
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.integrations.dask import get_worker_context
from rapidsmpf.streaming.core.context import Context

import polars as pl

from cudf_polars.experimental.dask_registers import DaskRegisterManager

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from distributed import Client
    from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo, StatsCollector
    from cudf_polars.experimental.parallel import ConfigOptions


class EvaluatePipelineCallback(Protocol):
    """Protocol for the evaluate_pipeline callback."""

    def __call__(
        self,
        ir: IR,
        partition_info: MutableMapping[IR, PartitionInfo],
        config_options: ConfigOptions,
        stats: StatsCollector,
        collective_id_map: dict[IR, list[int]],
        rmpf_context: Context | None = None,
        *,
        collect_metadata: bool = False,
    ) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
        """Evaluate a pipeline and return the result DataFrame and metadata."""
        ...


def get_dask_client() -> Client:
    """Get a distributed Dask client."""
    client = get_client()
    DaskRegisterManager.register_once()
    DaskRegisterManager.run_on_cluster(client)
    return client


def evaluate_pipeline_dask(
    callback: EvaluatePipelineCallback,
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
    stats: StatsCollector,
    collective_id_map: dict[IR, list[int]],
    *,
    collect_metadata: bool = False,
) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Evaluate a RapidsMPF streaming pipeline on a Dask cluster.

    Parameters
    ----------
    callback
        The callback function to evaluate the pipeline.
    ir
        The IR node.
    partition_info
        The partition information.
    config_options
        The configuration options.
    stats
        The statistics collector.
    collective_id_map
        Mapping from Shuffle/Repartition/Join IR nodes to reserved collective IDs.
    collect_metadata
        Whether to collect metadata.

    Returns
    -------
    The output DataFrame and metadata collector.
    """
    client = get_dask_client()
    result = client.run(
        _evaluate_pipeline_dask,
        callback,
        ir,
        partition_info,
        config_options,
        stats,
        collective_id_map,
        collect_metadata=collect_metadata,
    )
    dfs: list[pl.DataFrame] = []
    metadata_collector: list[ChannelMetadata] = []
    for df, md in result.values():
        dfs.append(df)
        if md is not None:
            metadata_collector.extend(md)

    return pl.concat(dfs), metadata_collector or None


def _evaluate_pipeline_dask(
    callback: EvaluatePipelineCallback,
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
    stats: StatsCollector,
    collective_id_map: dict[IR, list[int]],
    dask_worker: Any = None,
    *,
    collect_metadata: bool = False,
) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Build and evaluate a RapidsMPF streaming pipeline.

    Parameters
    ----------
    callback
        The callback function to evaluate the pipeline.
    ir
        The IR node.
    partition_info
        The partition information.
    config_options
        The configuration options.
    stats
        The statistics collector.
    collective_id_map
        Mapping from Shuffle/Repartition/Join IR nodes to reserved collective IDs.
    dask_worker
        Dask worker reference.
        This kwarg is automatically populated by Dask
        when evaluate_pipeline is called with `client.run`.
    collect_metadata
        Whether to collect metadata.

    Returns
    -------
    The output DataFrame and metadata collector.
    """
    assert dask_worker is not None, "Dask worker must be provided"
    assert config_options.executor.name == "streaming", "Executor must be streaming"

    # NOTE: The Dask-CUDA cluster must be bootstrapped
    # ahead of time using bootstrap_dask_cluster
    # (rapidsmpf.integrations.dask.bootstrap_dask_cluster).
    # TODO: Automatically bootstrap the cluster if necessary.
    options = Options(
        {"num_streaming_threads": str(max(config_options.executor.max_io_threads, 1))}
        | get_environment_variables()
    )
    dask_context = get_worker_context(dask_worker)
    with Context(
        dask_context.comm, dask_context.br, options, dask_context.statistics
    ) as rmpf_context:
        # IDs are already reserved by the caller, just pass them through
        return callback(
            ir,
            partition_info,
            config_options,
            stats,
            collective_id_map,
            rmpf_context,
            collect_metadata=collect_metadata,
        )
