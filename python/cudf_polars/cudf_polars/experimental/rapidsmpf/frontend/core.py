# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU frontend core."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import socket
from typing import TYPE_CHECKING, Any, Self, TypeVar

import cuda.core
from rapidsmpf.coll import AllGather
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.statistics import Statistics
from rapidsmpf.streaming.core.actor import run_actor_network
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import polars as pl

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IRExecutionContext
from cudf_polars.experimental.base import StatsCollector
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.experimental.rapidsmpf.collectives import ReserveOpIDs
from cudf_polars.experimental.rapidsmpf.collectives.common import reserve_op_id
from cudf_polars.experimental.rapidsmpf.core import generate_network
from cudf_polars.experimental.rapidsmpf.tracing import log_query_plan
from cudf_polars.experimental.rapidsmpf.utils import empty_table_chunk
from cudf_polars.experimental.statistics import collect_statistics
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping
    from concurrent.futures import ThreadPoolExecutor

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo
    from cudf_polars.experimental.parallel import ConfigOptions
    from cudf_polars.utils.config import StreamingExecutor


T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class ClusterInfo:
    """
    Diagnostic information about a single rank in the cluster.

    Attributes
    ----------
    pid
        Process ID of the current rank.
    hostname
        Hostname of the machine running this rank.
    cuda_visible_devices
        Value of ``CUDA_VISIBLE_DEVICES``, or ``None`` if unset.
    gpu_uuid
        UUID of the current CUDA device.
    """

    pid: int
    hostname: str
    cuda_visible_devices: str | None
    gpu_uuid: str

    @classmethod
    def local(cls) -> ClusterInfo:
        """
        Build a :class:`ClusterInfo` for the current process and GPU.

        Returns
        -------
        Diagnostic information for this rank.
        """
        return cls(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            gpu_uuid=cuda.core.Device().uuid,
        )


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
        if nranks > 1 and engine_options.get("allow_gpu_sharing", False) is False:
            uuids = [info.gpu_uuid for info in self.gather_cluster_info()]
            if len(uuids) != len(set(uuids)):
                raise RuntimeError(
                    "Multiple ranks share the same GPU (UUID collision detected). "
                    f"UUIDs: {uuids}. Set allow_gpu_sharing=True to allow this."
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

    def gather_cluster_info(self) -> list[ClusterInfo]:
        """
        Collect diagnostic information from every rank.

        Returns
        -------
        List of :class:`ClusterInfo`, one per rank.
        """
        raise NotImplementedError

    def gather_statistics(self, *, clear: bool = False) -> list[Statistics]:
        """
        Collect statistics from every rank.

        Parameters
        ----------
        clear
            If ``True``, clear each rank's statistics after gathering.

        Returns
        -------
        List of :class:`~rapidsmpf.statistics.Statistics`, one per rank,
        ordered by rank index.
        """
        raise NotImplementedError

    def global_statistics(self, *, clear: bool = False) -> Statistics:
        """
        Collect statistics from every rank and merge them into a single global statistics.

        Parameters
        ----------
        clear
            If ``True``, clear each rank's statistics after gathering.

        Returns
        -------
        A merged :class:`~rapidsmpf.statistics.Statistics`: per-stat counts
        and values are summed, maxima are reduced with ``max``. Formatters
        are taken from rank 0.
        """
        return Statistics.merge(self.gather_statistics(clear=clear))

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

    def _run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> list[T]:
        """
        Execute a function on all ranks.

        Parameters
        ----------
        func
            Function to execute.
        args
            Arguments to pass to the function.
        kwargs
            Keyword arguments to pass to the function.

        Returns
        -------
        List of results from calling ``func``, one per rank.
        """
        raise NotImplementedError


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


def all_gather_host_data(
    comm: Communicator,
    br: BufferResource,
    op_id: int,
    data: bytes | bytearray,
) -> list[bytes]:
    """
    Gather host data from every rank using an AllGather collective.

    Each rank contributes a buffer of host bytes; every rank receives back
    an ordered list containing the contributions from all ranks (index `i`
    holds the bytes sent by rank `i`).

    This function is blocking: all ranks must call it, and each rank
    waits until the collective completes. The input buffer is copied
    and cannot be stream-ordered.

    Parameters
    ----------
    comm
        The communicator shared by all participating ranks.
    br
        Buffer resource for memory allocation.
    op_id
        Unique operation identifier for this collective.
    data
        Host-side buffer to broadcast from this rank.  Accepts any object
        that implements the buffer protocol (``bytes``, ``bytearray``,
        ``memoryview``, etc.).

    Returns
    -------
    List of bytes, one element per rank, ordered by rank index.
    """
    allgather = AllGather(
        comm=comm,
        op_id=op_id,
        br=br,
        statistics=Statistics(enable=False),
    )
    allgather.insert(0, PackedData.from_host_bytes(data, br))
    allgather.insert_finished()
    results = allgather.wait_and_extract(ordered=True)
    return [r.to_host_bytes() for r in results]


def allgather_stats(
    comm: Communicator,
    br: BufferResource,
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
) -> StatsCollector:
    """
    Collect scan statistics on rank 0 and distribute to all ranks.

    When ``comm.nranks == 1`` the allgather is skipped and statistics are
    collected locally.

    Parameters
    ----------
    comm
        Communicator shared by all participating ranks.
    br
        Buffer resource for the allgather allocation.
    ir
        Root of the pre-lowered IR graph (same object on every rank).
    config_options
        Executor configuration.

    Returns
    -------
    A :class:`StatsCollector` valid for the local rank's IR node objects.
    """
    if comm.nranks == 1:
        return collect_statistics(ir, config_options)

    if comm.rank == 0:
        stats = collect_statistics(ir, config_options)
        data = json.dumps(stats.serialize(ir)).encode()
    else:
        data = b""

    with reserve_op_id() as op_id:
        all_data = all_gather_host_data(comm, br, op_id, data)

    if comm.rank == 0:
        return stats
    return StatsCollector.deserialize(json.loads(all_data[0]), ir)


def evaluate_on_rank(
    ctx: Context,
    comm: Communicator,
    py_executor: ThreadPoolExecutor,
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    *,
    collect_metadata: bool = False,
) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Evaluate a polars IR plan on a single rank.

    This is the main worker-side entry point for multi-rank execution.
    It performs the following steps collectively across all ranks:

    1. Collect statistics (on rank 0 and allgather)
    2. Lower the IR graph
    3. Reserve collective operation IDs
    4. Execute the lowered pipeline

    Parameters
    ----------
    ctx
        The active RapidsMPF streaming context for this rank.
    comm
        The active RapidsMPF communicator for this rank.
    py_executor
        Thread-pool executor used to drive the actor network.
    ir
        Root of the **pre-lowered** IR graph.
    config_options
        Executor configuration forwarded from the client.
    collect_metadata
        Whether to collect channel metadata during execution.

    Returns
    -------
    result
        This rank's output fragment as a Polars DataFrame.
    metadata
        Collected channel metadata if *collect_metadata* is ``True``,
        otherwise ``None``.
    """
    stats = allgather_stats(comm, ctx.br(), ir, config_options)
    ir, partition_info = lower_ir_graph(ir, config_options, stats)

    if comm.rank == 0:
        # At least for now, the query plan is identical on all ranks,
        # so we only log it once.
        log_query_plan(ir, config_options)

    with ReserveOpIDs(ir, config_options) as collective_id_map:
        return execute_ir_on_rank(
            ctx,
            comm,
            py_executor,
            ir,
            partition_info,
            config_options,
            stats,
            collective_id_map,
            collect_metadata=collect_metadata,
        )
