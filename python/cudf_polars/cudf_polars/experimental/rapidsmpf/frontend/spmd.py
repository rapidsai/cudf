# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""RapidsMPF streaming-engine using the SPMD Cluster style."""

from __future__ import annotations

import contextlib
import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

from rapidsmpf import bootstrap
from rapidsmpf.coll import AllGather
from rapidsmpf.communicator.single import (
    new_communicator as single_communicator,
)
from rapidsmpf.communicator.ucxx import barrier
from rapidsmpf.integrations.cudf.partition import unpack_and_concat
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.statistics import Statistics
from rapidsmpf.streaming.core.context import Context

import polars as pl

import pylibcudf as plc
import rmm.mr
from pylibcudf.contiguous_split import pack

from cudf_polars.experimental.rapidsmpf.collectives.common import reserve_op_id
from cudf_polars.experimental.rapidsmpf.frontend.core import (
    ClusterInfo,
    StreamingEngine,
    all_gather_host_data,
    check_reserved_keys,
    evaluate_on_rank,
    resolve_rapidsmpf_options,
)
from cudf_polars.experimental.rapidsmpf.frontend.hardware_binding import (
    HardwareBindingPolicy,
    bind_to_gpu,
)
from cudf_polars.experimental.rapidsmpf.utils import set_memory_resource
from cudf_polars.utils.config import SPMDContext

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.config import Options
    from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.parallel import ConfigOptions
    from cudf_polars.experimental.rapidsmpf.frontend.core import T
    from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
    from cudf_polars.utils.config import MemoryResourceConfig, StreamingExecutor


def evaluate_pipeline_spmd_mode(
    ir: IR,
    config_options: ConfigOptions[StreamingExecutor],
    *,
    collect_metadata: bool = False,
    query_id: uuid.UUID,
) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Build and evaluate a RapidsMPF streaming pipeline in SPMD mode.

    In SPMD mode every rank executes the same Python/Polars script
    independently.  Each rank owns its local DataFrames, which are
    treated as rank-local fragments of a larger distributed dataset and
    fed directly into the pipeline.  Collective operations (shuffles,
    all-gathers, etc.) coordinate across ranks to produce a globally
    consistent result.

    IR lowering is performed collectively on the workers: rank 0
    collects scan statistics and allgathers them, then every rank
    lowers the graph independently.

    Parameters
    ----------
    ir
        The pre-lowered IR node.
    config_options
        Executor configuration, including the rapidsmpf context and the
        Python thread-pool executor used to drive the actor network.
    collect_metadata
        Whether to collect runtime metadata.
    query_id
        A unique identifier for the query.

    Returns
    -------
    The concatenated output DataFrame and, if ``collect_metadata`` is
    True, the list of channel metadata objects; otherwise ``None``.
    """
    if config_options.executor.runtime != "rapidsmpf":
        raise RuntimeError("Runtime must be rapidsmpf")
    if config_options.executor.spmd_context is None:
        raise RuntimeError("spmd_context must be set for SPMD mode")
    comm = config_options.executor.spmd_context.comm
    context = config_options.executor.spmd_context.context
    py_executor = config_options.executor.spmd_context.py_executor

    return evaluate_on_rank(
        context,
        comm,
        py_executor,
        ir,
        config_options,
        collect_metadata=collect_metadata,
    )


def allgather_polars_dataframe(
    *,
    engine: SPMDEngine,
    local_df: pl.DataFrame,
    op_id: int,
) -> pl.DataFrame:
    """
    AllGather a rank-local DataFrame so every rank receives the full result.

    Each rank contributes its local ``local_df`` fragment and receives the
    concatenation of all ranks' fragments in rank order. This is the SPMD
    equivalent of a distributed ``collect``: after the call, every rank holds
    the same complete dataset.

    Parameters
    ----------
    engine
        The active :class:`SPMDEngine`.
    local_df
        Rank-local DataFrame to contribute.
    op_id
        Operation ID for this AllGather collective. Must be identical on every
        rank. For example, use :func:`reserve_op_id` to obtain a collision-free
        ID from the same pool used internally by cudf-polars. Avoid passing
        hardcoded integers.

    Returns
    -------
    DataFrame containing rows from all ranks, ordered by rank.

    Raises
    ------
    RuntimeError
        If ``engine`` has already been shut down.
    """
    comm = engine.comm
    ctx = engine.context
    stream = ctx.get_stream_from_pool()
    col_names = local_df.columns

    plc_table = plc.Table.from_arrow(local_df.to_arrow())

    packed_data = PackedData.from_cudf_packed_columns(
        pack(plc_table, stream),
        stream,
        ctx.br(),
    )

    # Bulk AllGather: each rank contributes once (sequence_number=0)
    allgather = AllGather(comm, op_id, ctx.br())
    try:
        allgather.insert(0, packed_data)
    finally:
        allgather.insert_finished()
    results = allgather.wait_and_extract(ordered=True)

    # Deserialize and concatenate each rank's contribution
    plc_result = unpack_and_concat(results, stream, ctx.br())

    # pylibcudf Table -> pl.DataFrame (restore column names)
    ret = pl.from_arrow(plc_result.to_arrow(col_names))
    assert isinstance(ret, pl.DataFrame)
    return ret


class SPMDEngine(StreamingEngine):
    """
    Multi-GPU Polars engine for SPMD executions.

    Bootstraps a RapidsMPF SPMD context and returns a matching engine.

    **SPMD execution model**

    SPMD (Single Program, Multiple Data) is a parallel programming model where each
    process runs the *same* Python script independently on its own slice of data.
    When launched with the RapidsMPF launcher `rrun`, multiple identical processes
    are started. Each process owns a rank-local :class:`~polars.LazyFrame`
    representing its fragment of the distributed dataset. Collective operations,
    such as shuffles, all-gathers, and joins, coordinate across ranks to produce
    a globally consistent result.

    Prefer :meth:`from_options` for typical use — pass a
    :class:`~cudf_polars.experimental.rapidsmpf.frontend.options.StreamingOptions`
    instance for a unified, typed interface. The ``__init__`` parameters
    (``rapidsmpf_options``, ``executor_options``, ``engine_options``) are
    intended for advanced use when fine-grained control is needed.

    This class is the primary entry point for SPMD execution. It:

    - Bootstraps a communicator connecting all ranks. When launched with ``rrun``
      this is a full UCXX communicator. When running as a normal single Python
      process (no ``rrun``) it falls back to a lightweight single-rank communicator
      that requires no external communication library (no UCXX, Ray, or Dask).
    - Creates a RapidsMPF :class:`~rapidsmpf.streaming.core.context.Context`
      that owns GPU memory and a CUDA-stream pool.

    All resources (communicator, stream pool, thread-pool) are released when
    :meth:`~SPMDEngine.shutdown` is called or the engine is used as a context
    manager.

    **Memory resource**

    ``SPMDEngine`` captures ``rmm.mr.get_current_device_resource()`` at construction,
    wraps it in ``RmmResourceAdaptor`` (so libcudf temporary allocations and the
    RapidsMPF ``Context`` share the same resource), sets the wrapped resource as
    current, and restores the original on shutdown.

    To use a custom allocator, call ``rmm.mr.set_current_device_resource(your_mr)``
    before constructing ``SPMDEngine``. Do not pre-wrap it in ``RmmResourceAdaptor``.

    .. code-block:: python

        import rmm

        # Optional: install a pool allocator before constructing SPMDEngine.
        # rmm.mr.set_current_device_resource(
        #     rmm.mr.PoolMemoryResource(rmm.mr.CudaMemoryResource())
        # )
        with SPMDEngine(...) as engine:
            ...

    **DataFrame and LazyFrame semantics**

    Because every rank runs an independent Python process, a :class:`~polars.DataFrame`
    is always *rank-local* i.e. it contains only that rank's fragment of the distributed
    dataset.  This is true whether the DataFrame originates from a file reader or from
    Python literals.

    File-based sources (``scan_parquet``, ``scan_csv``, ...) distribute their work
    automatically: the engine assigns disjoint file- or row-group ranges to each rank,
    so different ranks produce different data.

    An in-memory ``DataFrame`` (or one produced by a previous ``collect``) is already
    rank-local by construction.  Each rank processes its own copy in full; the engine
    does **not** re-slice it across ranks.  In particular, the two patterns below are
    equivalent:

    .. code-block:: python

        # One-step: scan and transform in a single pipeline
        result = pl.scan_parquet(...).pipe(transform).collect(engine=engine)

        # Two-step: collect an intermediate result, then transform
        intermediate = pl.scan_parquet(...).collect(engine=engine)
        result = intermediate.lazy().pipe(transform).collect(engine=engine)

    In both cases rank k operates on exactly the data it read from parquet. The
    intermediate ``collect`` simply materializes the data in memory; it does not
    change which rows belong to which rank.

    **Query symmetry requirement**

    Every rank must issue the *same* sequence of Polars queries in the *same*
    order.  Collective operations (shuffles, all-gathers, joins) are matched
    across ranks by a monotonically increasing operation ID — if one rank calls
    a collective that another rank does not, all ranks will deadlock.  This means
    your driver script must be fully deterministic: avoid rank-conditional
    ``collect`` calls, early exits, or any branching that would cause different
    ranks to execute different query graphs.

    Parameters
    ----------
    comm
        An already-bootstrapped communicator. When provided, the bootstrap step
        is skipped and the caller retains ownership; the communicator is **not**
        closed on shutdown. Pass this to share a single communicator across multiple
        engine lifetimes (e.g. a session-scoped pytest fixture).
        When ``None`` (default) a new communicator is bootstrapped automatically.
    rapidsmpf_options
        RapidsMPF-specific options. Defaults to the reading ``RAPIDSMPF_*``
        environment variables.
    executor_options
        Executor-specific options (e.g. ``max_rows_per_partition``).
    engine_options
        Engine-specific keyword arguments (e.g. ``raise_on_fail``,
        ``parquet_options``).

    Raises
    ------
    TypeError
        If ``executor_options`` or ``engine_options`` contains a reserved key.

    Notes
    -----
    Calls
    :func:`~cudf_polars.experimental.rapidsmpf.frontend.hardware_binding.bind_to_gpu`
    at construction time, before RMM and communicator initialisation, so that
    CPU affinity, NUMA memory policy, and ``UCX_NET_DEVICES`` are set as early
    as possible.  By default, binding is skipped under ``rrun`` (which already
    performs its own binding) — see
    :attr:`HardwareBindingPolicy.skip_under_rrun`.

    Examples
    --------
    Context-manager style (recommended for scripts):

    >>> with SPMDEngine() as engine:  # doctest: +SKIP
    ...     result = (
    ...         df.lazy().group_by("a").agg(pl.col("b").sum()).collect(engine=engine)
    ...     )
    ...     full = allgather_polars_dataframe(engine=engine, local_df=result, op_id=0)

    Direct style (Jupyter / long-lived clusters):

    >>> engine = SPMDEngine()  # doctest: +SKIP
    >>> result = df.lazy().collect(engine=engine)  # doctest: +SKIP
    >>> engine.shutdown()  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        comm: Communicator | None = None,
        rapidsmpf_options: Options | None = None,
        executor_options: dict[str, Any] | None = None,
        engine_options: dict[str, Any] | None = None,
    ) -> None:
        executor_options = executor_options or {}
        engine_options = engine_options or {}

        check_reserved_keys(executor_options, engine_options)
        hw_binding = cast(
            HardwareBindingPolicy,
            engine_options.get("hardware_binding", HardwareBindingPolicy()),
        )
        bind_to_gpu(hw_binding)

        rapidsmpf_options = resolve_rapidsmpf_options(rapidsmpf_options)
        mr_config: MemoryResourceConfig | None = engine_options.get(
            "memory_resource_config", None
        )
        base_mr = (
            mr_config.create_memory_resource()
            if mr_config is not None
            else rmm.mr.get_current_device_resource()
        )
        mr = RmmResourceAdaptor(base_mr)
        if comm is None:
            if bootstrap.is_running_with_rrun():
                comm = bootstrap.create_ucxx_comm(
                    progress_thread=ProgressThread(),
                    type=bootstrap.BackendType.AUTO,
                    options=rapidsmpf_options,
                )
            else:
                comm = single_communicator(
                    progress_thread=ProgressThread(),
                    options=rapidsmpf_options,
                )
        # else: caller-provided comm; the caller retains ownership

        self._py_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=cast(int, executor_options.get("num_py_executors", 8)),
            thread_name_prefix="spmd-executor",
        )
        self._mr: RmmResourceAdaptor = mr
        exit_stack = contextlib.ExitStack()
        try:
            exit_stack.callback(self._py_executor.shutdown, wait=False)
            exit_stack.enter_context(set_memory_resource(mr))
            # ``Context`` is *not* registered as a context manager so that
            # :meth:`_reset` can swap it mid-life without leaving the
            # exit-stack holding a stale reference. ``_cleanup_ctx`` is
            # registered instead — it shuts down whatever ``self._ctx`` is
            # at engine-shutdown time (i.e. the latest reset's Context).
            ctx = Context.from_options(comm.logger, mr, rapidsmpf_options)
            exit_stack.callback(self._cleanup_ctx)
            self._comm: Communicator | None = comm
            self._ctx: Context | None = ctx
            super().__init__(
                nranks=comm.nranks,
                executor_options={
                    **executor_options,
                    "runtime": "rapidsmpf",
                    "cluster": "spmd",
                    "spmd_context": SPMDContext(
                        comm=comm, context=ctx, py_executor=self._py_executor
                    ),
                },
                engine_options={
                    **engine_options,
                    "memory_resource": ctx.br().device_mr,
                },
                exit_stack=exit_stack,
            )
        except Exception:
            exit_stack.close()
            raise

    def _cleanup_ctx(self) -> None:
        """
        Shut down the current ``self._ctx`` if any; called from exit-stack.

        ``Context.shutdown()`` is idempotent on the rapidsmpf C++ side, so this is
        safe even if a prior ``_reset`` already shut down a now-replaced Context.
        """
        if self._ctx is not None:
            self._ctx.shutdown()
            self._ctx = None

    @classmethod
    def from_options(cls, options: StreamingOptions) -> SPMDEngine:
        """
        Create an :class:`SPMDEngine` from a :class:`StreamingOptions` object.

        This is the recommended way to construct an ``SPMDEngine`` for typical
        use. All RapidsMPF, executor, and engine options are read from
        ``options``; unset fields fall back to environment variables and then
        to built-in defaults.

        Parameters
        ----------
        options
            Unified streaming configuration.

        Returns
        -------
        A new :class:`SPMDEngine` instance.

        Examples
        --------
        >>> from cudf_polars.experimental.rapidsmpf.frontend.options import (
        ...     StreamingOptions,
        ... )
        >>> opts = StreamingOptions(num_streaming_threads=8, fallback_mode="silent")
        >>> with SPMDEngine.from_options(opts) as engine:  # doctest: +SKIP
        ...     result = df.lazy().collect(engine=engine)
        """
        return cls(
            rapidsmpf_options=options.to_rapidsmpf_options(),
            executor_options=options.to_executor_options(),
            engine_options=options.to_engine_options(),
        )

    def _reset(
        self,
        *,
        rapidsmpf_options: Options | None = None,
        executor_options: dict[str, Any] | None = None,
        engine_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Reset the engine; see :meth:`StreamingEngine._reset` for the contract.

        Must be called collectively on all ranks. A barrier ensures no
        rank tears down its Context while peers may still be using it.
        """
        if self._ctx is None:
            raise RuntimeError("Cannot reset a shut-down engine")
        assert self._comm is not None
        super()._reset(
            rapidsmpf_options=rapidsmpf_options,
            executor_options=executor_options,
            engine_options=engine_options,
        )
        executor_options = executor_options or {}
        engine_options = engine_options or {}
        rapidsmpf_options = resolve_rapidsmpf_options(rapidsmpf_options)

        # Collective: synchronize all ranks before tearing down the Context.
        if self._comm.nranks > 1:
            barrier(self._comm)
        # Same-thread shutdown, _reset runs on the thread that built the
        # Context (the test driver's main thread). The per-engine RMM
        # resource is kept alive across resets, see :meth:`_cleanup_ctx`.
        self._ctx.shutdown()
        self._ctx = Context.from_options(self._comm.logger, self._mr, rapidsmpf_options)

        # Re-run ``StreamingEngine.__init__`` on the existing instance to
        # reconfigure the polars ``GPUEngine`` layer (``self.config``,
        # ``self.device``, etc.) with the new options. Pass the existing
        # ``self._exit_stack`` so any registered callbacks (notably
        # ``_cleanup_ctx`` and ``set_memory_resource``) survive.
        StreamingEngine.__init__(
            self,
            nranks=self._comm.nranks,
            executor_options={
                **executor_options,
                "runtime": "rapidsmpf",
                "cluster": "spmd",
                "spmd_context": SPMDContext(
                    comm=self._comm,
                    context=self._ctx,
                    py_executor=self._py_executor,
                ),
            },
            engine_options={
                **engine_options,
                "memory_resource": self._ctx.br().device_mr,
            },
            exit_stack=self._exit_stack,
        )

    @property
    def rank(self) -> int:
        """
        Rank index within the cluster (zero-based).

        Returns
        -------
        Rank index.

        Raises
        ------
        RuntimeError
            If called after :meth:`shutdown`.
        """
        return self.comm.rank

    @property
    def comm(self) -> Communicator:
        """
        The active RapidsMPF communicator.

        Returns
        -------
        Active communicator.

        Raises
        ------
        RuntimeError
            If called after :meth:`shutdown`.
        """
        if self._comm is None:
            raise RuntimeError("comm is not available after shutdown")
        return self._comm

    @property
    def context(self) -> Context:
        """
        The active RapidsMPF streaming context.

        Returns
        -------
        Active streaming context.

        Raises
        ------
        RuntimeError
            If called after :meth:`shutdown`.
        """
        if self._ctx is None:
            raise RuntimeError("context is not available after shutdown")
        return self._ctx

    def gather_cluster_info(self) -> list[ClusterInfo]:
        """
        Collect diagnostic information from every rank.

        This is a collective operation, every rank must call it.

        Returns
        -------
        List of :class:`ClusterInfo`, one per rank.
        """
        data = json.dumps(dataclasses.asdict(ClusterInfo.local())).encode()
        with reserve_op_id() as op_id:
            results = all_gather_host_data(self.comm, self.context.br(), op_id, data)
        return [ClusterInfo(**json.loads(r)) for r in results]

    def gather_statistics(self, *, clear: bool = False) -> list[Statistics]:
        """
        Collect statistics from every rank via an all-gather.

        This is a collective operation, every rank must call it.

        Parameters
        ----------
        clear
            If ``True``, clear each rank's statistics after gathering.

        Returns
        -------
        List of :class:`~rapidsmpf.statistics.Statistics`, one per rank,
        ordered by rank index.
        """
        # Serialize before the optional clear so the returned stats still carry data.
        data = self.context.statistics().serialize()
        with reserve_op_id() as op_id:
            results = all_gather_host_data(self.comm, self.context.br(), op_id, data)
        if clear:
            self.context.statistics().clear()
        return [Statistics.deserialize(r) for r in results]

    def shutdown(self) -> None:
        """
        Shut down the engine and release all owned resources.

        Idempotent: safe to call more than once. Must be called on the same
        thread that created the engine.
        """
        if self._ctx is None:
            return  # already shut down

        # Order matters: ``super().shutdown()`` closes ``self._exit_stack``,
        # which invokes ``self._cleanup_ctx``. That requires ``self._ctx`` to
        # still be set so the rapidsmpf Context can be shut down correctly.
        # Clear the references only after shutdown completes.
        super().shutdown()
        self._comm = None
        self._ctx = None

    def _run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> list[T]:
        data = json.dumps(func(*args, **kwargs)).encode()
        with reserve_op_id() as op_id:
            results = all_gather_host_data(self.comm, self.context.br(), op_id, data)

        return [json.loads(r) for r in results]
