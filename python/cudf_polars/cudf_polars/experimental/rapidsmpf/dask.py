# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Dask-based execution with the streaming RapidsMPF runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from distributed import get_client
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.integrations.dask import get_worker_context
from rapidsmpf.streaming.core.context import Context

import polars as pl

from cudf_polars.experimental.dask_registers import DaskRegisterManager

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping

    from distributed import Client

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo, StatsCollector
    from cudf_polars.experimental.parallel import ConfigOptions


def get_dask_client() -> Client:
    """Get a distributed Dask client."""
    client = get_client()
    DaskRegisterManager.register_once()
    DaskRegisterManager.run_on_cluster(client)
    return client


def evaluate_pipeline_dask(
    callback: Callable[
        [
            IR,
            MutableMapping[IR, PartitionInfo],
            ConfigOptions,
            StatsCollector,
            dict[IR, int],
            Context | None,
        ],
        pl.DataFrame,
    ],
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
    stats: StatsCollector,
    shuffle_id_map: dict[IR, int],
) -> pl.DataFrame:
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
    shuffle_id_map
        Mapping from Shuffle/Repartition IR nodes to pre-allocated shuffle IDs.

    Returns
    -------
    The output DataFrame.
    """
    client = get_dask_client()
    result = client.run(
        _evaluate_pipeline_dask,
        callback,
        ir,
        partition_info,
        config_options,
        stats,
        shuffle_id_map,
    )
    return pl.concat(result.values())


def _evaluate_pipeline_dask(
    callback: Callable[
        [
            IR,
            MutableMapping[IR, PartitionInfo],
            ConfigOptions,
            StatsCollector,
            dict[IR, int],
            Context | None,
        ],
        pl.DataFrame,
    ],
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions,
    stats: StatsCollector,
    shuffle_id_map: dict[IR, int],
    dask_worker: Any = None,
) -> pl.DataFrame:
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
    shuffle_id_map
        Mapping from Shuffle/Repartition IR nodes to pre-allocated shuffle IDs.
    dask_worker
        Dask worker reference.
        This kwarg is automatically populated by Dask
        when evaluate_pipeline is called with `client.run`.

    Returns
    -------
    The output DataFrame.
    """
    options = Options(get_environment_variables())

    assert dask_worker is not None, "Dask worker must be provided"
    # NOTE: The Dask-CUDA cluster must be bootstrapped
    # ahead of time using bootstrap_dask_cluster
    # (rapidsmpf.integrations.dask.bootstrap_dask_cluster).
    # TODO: Automatically bootstrap the cluster if necessary.

    dask_context = get_worker_context(dask_worker)
    rmpf_context = Context(dask_context.comm, dask_context.br, options)
    return callback(
        ir, partition_info, config_options, stats, shuffle_id_map, rmpf_context
    )
