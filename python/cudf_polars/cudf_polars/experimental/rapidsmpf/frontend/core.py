# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU frontend core."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import socket
import uuid
from typing import TYPE_CHECKING, Any, Self, TypeVar

import cuda.core
from rapidsmpf.coll import AllGather
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.statistics import Statistics
from rapidsmpf.streaming.core.actor import run_actor_network
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import polars as pl

import cudf_polars.quent._logging
from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IRExecutionContext
from cudf_polars.experimental.base import StatsCollector
from cudf_polars.experimental.parallel import lower_ir_graph_with_node_map
from cudf_polars.experimental.rapidsmpf.collectives import ReserveOpIDs
from cudf_polars.experimental.rapidsmpf.collectives.common import reserve_op_id
from cudf_polars.experimental.rapidsmpf.core import generate_network
from cudf_polars.experimental.rapidsmpf.tracing import log_query_plan
from cudf_polars.experimental.rapidsmpf.utils import empty_table_chunk
from cudf_polars.experimental.statistics import collect_statistics
from cudf_polars.experimental.utils import _concat
from cudf_polars.quent._plan import build_plan

if TYPE_CHECKING:
    from collections.abc import Callable, MutableMapping
    from concurrent.futures import ThreadPoolExecutor

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.memory.buffer_resource import BufferResource
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata

    import cudf_polars.quent
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

    _quent_logger: cudf_polars.quent._logging.QuentLogger

    def __init__(
        self,
        *,
        nranks: int,
        executor_options: dict[str, Any],
        engine_options: dict[str, Any],
        exit_stack: contextlib.ExitStack | None = None,
    ):
        self._nranks = nranks
        self._quent_events: list[dict[str, Any]] = []  # populated on shutdown
        self._exit_stack: contextlib.ExitStack | None = (
            exit_stack or contextlib.ExitStack()
        )
        # allow_gpu_sharing is consumed here since polars' GPUEngine doesn't
        # accept it.
        engine_options = dict(engine_options)
        allow_gpu_sharing = engine_options.pop("allow_gpu_sharing", False)
        super().__init__(
            executor="streaming",
            executor_options=executor_options,
            **engine_options,
        )
        if nranks > 1 and not allow_gpu_sharing:
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

        Subclasses should emit their final lifecycle events and extend
        ``_worker_quent_events`` with remote worker events *before*
        calling ``super().shutdown()``. This base implementation drains
        any locally buffered Quent events (e.g. Engine init/exit emitted
        on the driver).

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

    # @property
    # def quent_events(self) -> list[dict[str, Any]]:
    #     """Return all Quent telemetry events collected during the engine's lifecycle."""
    #     import cudf_polars.quent._logging

    #     # TODO: this global log_buffer is messing things up.
    #     # This whole things needs to be rewritten.
    #     # we need to clear it at some point
    #     # but right now this function isn't idempotent; the second call won't include events from the client.
    #     events = []
    #     with cudf_polars.quent._logging.buffer_lock:
    #         events = list(cudf_polars.quent._logging.log_buffer)
    #         cudf_polars.quent._logging.log_buffer.clear()

    #     events.extend(self.worker_quent_events)
    #     return [x["event"] for x in sorted(events, key=lambda e: e.get("timestamp", 0))]

    # @property
    # def worker_quent_events(self) -> list[dict[str, Any]]:
    #     """
    #     Quent telemetry events collected from workers during shutdown.

    #     Empty until :meth:`shutdown` is called. For distributed engines
    #     (Ray, Dask) this contains events that were buffered on remote
    #     workers and drained back to the driver. Events are sorted by
    #     timestamp.

    #     Returns
    #     -------
    #     List of serialized Quent event dicts, sorted by timestamp.
    #     """
    #     return sorted(self._worker_quent_events, key=lambda e: e.get("timestamp", 0))

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
    worker_id: uuid.UUID,
    quent_context: cudf_polars.quent.QuentContext,
    quent_logger: cudf_polars.quent._logging.QuentLogger,
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
    worker_id
        Quent worker ID for this rank. When provided, rank 0 emits
        plan-level Quent telemetry (logical and physical plan
        declarations). Passed explicitly so that all frontends
        (SPMD, Ray, Dask) can supply their own ID without requiring
        a specific context type.
    quent_context
        The quent context to use for this query.
    quent_logger
        The logger collecting Quent events for this rank.

    Returns
    -------
    result
        This rank's output fragment as a Polars DataFrame.
    metadata
        Collected channel metadata if *collect_metadata* is ``True``,
        otherwise ``None``.
    """
    stats = allgather_stats(comm, ctx.br(), ir, config_options)

    # TODO: this logical_plan_id is probably wrong. We need this ID to summarize the *entire* logical plan.
    # not just the last node. We should just generate a random ID on the client and
    # pass that in.
    logical_plan_id = uuid.UUID(int=ir.get_stable_id())
    physical_plan_id = uuid.uuid4()

    # # TODO: Make context a Union[SPMDContext, RayContext, DaskContext]
    # streaming_context = (
    #     config_options.executor.spmd_context
    #     or config_options.executor.ray_context
    #     or config_options.executor.dask_context
    # )
    # assert streaming_context is not None, (
    #     f"No streaming context provided, worker_id={worker_id}"
    # )

    if worker_id is not None:
        # TODO: split out build from emit.
        plan, ops, ports, logical_op_by_id = build_plan(
            ir,
            config_options,
            query_id=quent_context.query.id,
            plan_id=logical_plan_id,
            worker_id=worker_id,
            instance_name="logical",
            parent_plan_id=None,
            parent_operators_by_node_id=None,
        )
        if comm.rank == 0:
            quent_context.emit_plan_declarations(quent_logger, plan, ops, ports)

    ir, partition_info, node_map = lower_ir_graph_with_node_map(
        ir, config_options, stats
    )

    if worker_id is not None:
        # Problem: each worker gets their own `plan.id` so we can't
        # associate the physical plan with the logical plan on any
        # rank other than 0.
        # We would need to pass the plan ID in along with the IR...
        log_query_plan(ir, config_options)
        quent_context.emit_physical_plan_events(
            quent_logger,
            ir,
            config_options,
            plan_id=physical_plan_id,
            worker_id=worker_id,
            parent_plan_id=logical_plan_id,
            node_map=node_map,
            logical_op_by_id=logical_op_by_id,
        )

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
