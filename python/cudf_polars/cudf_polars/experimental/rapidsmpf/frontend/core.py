# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU frontend core."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Self

from rapidsmpf.streaming.core.actor import run_actor_network
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import polars as pl

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IRExecutionContext
from cudf_polars.experimental.rapidsmpf.core import generate_network
from cudf_polars.experimental.rapidsmpf.utils import empty_table_chunk
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from concurrent.futures import ThreadPoolExecutor

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo, StatsCollector
    from cudf_polars.experimental.parallel import ConfigOptions
    from cudf_polars.utils.config import StreamingExecutor


def execute_ir_on_rank(
    ctx: Context,
    comm: Communicator,
    py_executor: ThreadPoolExecutor,
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions[StreamingExecutor],
    stats: StatsCollector,
    collective_id_map: dict[IR, list[int]],
    *,
    collect_metadata: bool = False,
) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Execute a Polars IR query on a single rank's GPU.

    Shared implementation used by the frontends. Each frontend acquires its local
    ``ctx``, ``comm``, and ``py_executor`` from its own per-rank state and delegates
    to this function for the actual execution.

    Parameters
    ----------
    ctx
        The active RapidsMPF streaming context for this rank.
    comm
        The active RapidsMPF communicator for this rank.
    py_executor
        Thread-pool executor used to drive the actor network.
    ir
        Root IR node describing the query.
    partition_info
        Per-node partition metadata.
    config_options
        Executor configuration forwarded from the client.
    stats
        Statistics collector.
    collective_id_map
        Mapping from IR nodes to their pre-allocated collective operation IDs.
    collect_metadata
        Whether to collect channel metadata during execution.

    Returns
    -------
    result
        This rank's output fragment as a Polars DataFrame.
    metadata
        Collected channel metadata if ``collect_metadata`` is ``True``,
        otherwise ``None``.
    """
    ir_context = IRExecutionContext(get_cuda_stream=ctx.get_stream_from_pool)
    metadata_collector: list[ChannelMetadata] | None = [] if collect_metadata else None

    nodes, output = generate_network(
        ctx,
        comm,
        ir,
        partition_info,
        config_options,
        stats,
        ir_context=ir_context,
        collective_id_map=collective_id_map,
        metadata_collector=metadata_collector,
    )

    run_actor_network(actors=nodes, py_executor=py_executor)

    messages = output.release()
    chunks = [
        TableChunk.from_message(msg, br=ctx.br()).make_available_and_spill(
            ctx.br(), allow_overbooking=True
        )
        for msg in messages
    ]
    if chunks:
        dfs = [
            DataFrame.from_table(
                chunk.table_view(),
                list(ir.schema.keys()),
                list(ir.schema.values()),
                chunk.stream,
            )
            for chunk in chunks
        ]
        df = _concat(*dfs, context=ir_context)
    else:
        stream = ir_context.get_cuda_stream()
        chunk = empty_table_chunk(ir, ctx, stream)
        df = DataFrame.from_table(
            chunk.table_view(),
            list(ir.schema.keys()),
            list(ir.schema.values()),
            stream,
        )
    return df.to_polars(), metadata_collector


_RESERVED_EXECUTOR_KEYS: frozenset[str] = frozenset(
    {"runtime", "cluster", "spmd_context", "ray_context", "dask_context"}
)
_RESERVED_ENGINE_KEYS: frozenset[str] = frozenset({"memory_resource", "executor"})


def check_reserved_keys(
    executor_options: dict[str, Any],
    engine_options: dict[str, Any],
) -> None:
    """
    Raise :exc:`TypeError` if any reserved keys are present in the option dicts.

    Parameters
    ----------
    executor_options
        Executor-specific options to validate.
    engine_options
        Engine-specific options to validate.

    Raises
    ------
    TypeError
        If ``executor_options`` contains any reserved key.
    TypeError
        If ``engine_options`` contains any reserved key.
    """
    if bad := _RESERVED_EXECUTOR_KEYS & executor_options.keys():
        raise TypeError(f"executor_options may not contain reserved keys: {bad}")
    if bad := _RESERVED_ENGINE_KEYS & engine_options.keys():
        raise TypeError(f"engine_options may not contain reserved keys: {bad}")


class StreamingEngine(pl.GPUEngine):
    """
    Base class for multi-GPU Polars engines.

    The engine manages the lifecycle of a streaming execution and can
    be used as a context manager. On exit, :meth:`shutdown` is called.

    Notes
    -----
    The engine must be created and shut down on the same thread. In particular,
    destruction and context manager exit must occur on the thread that created
    the instance.

    Parameters
    ----------
    nranks
        Number of ranks (workers or GPUs) in the cluster.
    executor_options
        Executor-specific options (e.g. ``max_rows_per_partition``).
    engine_options
        Engine-specific keyword arguments (e.g. ``raise_on_fail``,
        ``parquet_options``).
    exit_stack
        A :class:`contextlib.ExitStack` whose registered contexts are closed
        when :meth:`shutdown` is called. If ``None``, an empty stack is created.
    """

    def __init__(
        self,
        *,
        nranks: int,
        executor_options: dict[str, Any],
        engine_options: dict[str, Any],
        exit_stack: contextlib.ExitStack | None = None,
    ):
        self._nranks = nranks
        self._exit_stack: contextlib.ExitStack | None = (
            exit_stack or contextlib.ExitStack()
        )
        super().__init__(
            executor="streaming",
            executor_options=executor_options,
            **engine_options,
        )

    @property
    def nranks(self) -> int:
        """
        Number of ranks (for example GPUs or workers) in the cluster.

        Local execution without a cluster returns 1.

        Returns
        -------
        Number of ranks.
        """
        return self._nranks

    def shutdown(self) -> None:
        """
        Shut down engine and release all owned resources.

        Idempotent: safe to call more than once. Must be called on the same
        thread that created the engine.
        """
        if self._exit_stack is None:
            return  # already shut down
        try:
            self._exit_stack.close()
        finally:
            self._exit_stack = None
            self.device = None
            self.memory_resource = None
            self.config = {}

    def __enter__(self) -> Self:
        """Enter the context manager, returning ``self``."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit the context manager, calling :meth:`shutdown`."""
        self.shutdown()
