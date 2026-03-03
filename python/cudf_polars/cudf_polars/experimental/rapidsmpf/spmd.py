# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""RapidsMPF streaming-engine using the SPMD Cluster style."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

from rapidsmpf import bootstrap
from rapidsmpf.coll import AllGather
from rapidsmpf.config import Options, get_environment_variables
from rapidsmpf.integrations.cudf.partition import unpack_and_concat
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.core.actor import run_actor_network
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import polars as pl

import pylibcudf as plc
import rmm.mr
from pylibcudf.contiguous_split import pack

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IRExecutionContext
from cudf_polars.experimental.rapidsmpf.core import generate_network
from cudf_polars.experimental.rapidsmpf.utils import empty_table_chunk
from cudf_polars.experimental.utils import _concat
from cudf_polars.utils.config import ContextSPMD

if TYPE_CHECKING:
    from collections.abc import Iterator, MutableMapping

    from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo, StatsCollector
    from cudf_polars.experimental.parallel import ConfigOptions
    from cudf_polars.utils.config import StreamingExecutor


def evaluate_pipeline_spmd_mode(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions[StreamingExecutor],
    stats: StatsCollector,
    collective_id_map: dict[IR, list[int]],
    *,
    collect_metadata: bool = False,
) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Build and evaluate a RapidsMPF streaming pipeline in SPMD mode.

    In SPMD mode every rank executes the same Python/Polars script
    independently.  Each rank owns its local DataFrames, which are
    treated as rank-local fragments of a larger distributed dataset and
    fed directly into the pipeline.  Collective operations (shuffles,
    all-gathers, etc.) coordinate across ranks to produce a globally
    consistent result.

    Parameters
    ----------
    ir
        The IR node.
    partition_info
        The partition information.
    config_options
        Executor configuration, including the rapidsmpf context and the
        Python thread-pool executor used to drive the actor network.
    stats
        The statistics collector.
    collective_id_map
        Mapping from IR nodes to their pre-allocated collective operation
        IDs.
    collect_metadata
        Whether to collect runtime metadata.

    Returns
    -------
    The concatenated output DataFrame and, if ``collect_metadata`` is
    True, the list of channel metadata objects; otherwise ``None``.
    """
    if config_options.executor.runtime != "rapidsmpf":
        raise RuntimeError("Runtime must be rapidsmpf")
    if config_options.executor.spmd is None:
        raise RuntimeError("spmd must be set for SPMD mode")
    context = config_options.executor.spmd.context
    py_executor = config_options.executor.spmd.py_executor

    # Create the IR execution context
    ir_context = IRExecutionContext(get_cuda_stream=context.get_stream_from_pool)

    # Generate network nodes
    metadata_collector: list[ChannelMetadata] | None = [] if collect_metadata else None

    nodes, output = generate_network(
        context,
        ir,
        partition_info,
        config_options,
        stats,
        ir_context=ir_context,
        collective_id_map=collective_id_map,
        metadata_collector=metadata_collector,
    )

    # Run the network
    run_actor_network(actors=nodes, py_executor=py_executor)

    # Extract/return the concatenated result.
    # Keep chunks alive until after concatenation to prevent
    # use-after-free with stream-ordered allocations
    messages = output.release()
    chunks = [
        TableChunk.from_message(msg).make_available_and_spill(
            context.br(), allow_overbooking=True
        )
        for msg in messages
    ]
    dfs: list[DataFrame]
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
        # No chunks received - create an empty DataFrame with correct schema
        stream = ir_context.get_cuda_stream()
        chunk = empty_table_chunk(ir, context, stream)
        df = DataFrame.from_table(
            chunk.table_view(),
            list(ir.schema.keys()),
            list(ir.schema.values()),
            stream,
        )

    result = df.to_polars()
    df.stream.synchronize()
    return result, metadata_collector


def allgather_polars_dataframe(
    *,
    ctx: Context,
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
    ctx
        The RapidsMPF context.
    local_df
        Rank-local DataFrame to contribute.
    op_id
        Operation ID for this AllGather collective. Must be identical on every
        rank.

    Returns
    -------
    DataFrame containing rows from all ranks, ordered by rank.
    """
    stream = ctx.get_stream_from_pool()
    col_names = local_df.columns

    # pl.DataFrame -> pylibcudf Table (via Arrow, GPU transfer)
    plc_table = plc.Table.from_arrow(local_df.to_arrow())

    # Serialize for network transport
    packed_data = PackedData.from_cudf_packed_columns(
        pack(plc_table, stream),
        stream,
        ctx.br(),
    )

    # Bulk AllGather: each rank contributes once (sequence_number=0)
    allgather = AllGather(ctx.comm(), ctx.comm().progress_thread, op_id, ctx.br())
    allgather.insert(0, packed_data)
    allgather.insert_finished()
    results = allgather.wait_and_extract(ordered=True)

    # Deserialize and concatenate all ranks' contributions
    plc_result = unpack_and_concat(results, stream, ctx.br())

    # pylibcudf Table -> pl.DataFrame (restore column names)
    ret = pl.from_arrow(plc_result.to_arrow(col_names))
    assert isinstance(ret, pl.DataFrame)
    return ret


@contextmanager
def spmd_execution(
    *,
    executor_options: dict[str, object] | None = None,
    mr: rmm.mr.DeviceMemoryResource | None = None,
    options: Options | None = None,
    **engine_kwargs: Any,
) -> Iterator[tuple[Context, pl.GPUEngine]]:
    """
    Context manager that bootstraps a RapidsMPF SPMD context and a matching GPUEngine.

    **SPMD execution model**

    SPMD (Single Program, Multiple Data) is a parallel programming style where
    every process runs the *same* Python script independently on its own slice of
    data.  When you launch with ``rrun -n N pytest`` (or any ``rrun`` invocation),
    ``N`` identical processes are started.  Each process owns a rank-local
    :class:`~polars.LazyFrame` that represents its fragment of the distributed
    dataset.  Collective operations — shuffles, all-gathers, joins — coordinate
    across ranks so that the result is globally consistent.

    This context manager is the primary entry point for SPMD execution. It:

    - Bootstraps a UCXX communicator connecting all ``N`` ranks.
    - Creates a RapidsMPF :class:`~rapidsmpf.streaming.core.context.Context`
      that owns GPU memory and a CUDA-stream pool.
    - Returns a :class:`~polars.lazyframe.engine_config.GPUEngine` wired to that
      context so that ``LazyFrame.collect(engine=engine)`` dispatches through the
      RapidsMPF streaming executor.

    All resources (communicator, stream pool, thread-pool) are released on exit.

    **Query symmetry requirement**

    Every rank must issue the *same* sequence of Polars queries in the *same*
    order.  Collective operations (shuffles, all-gathers, joins) are matched
    across ranks by a monotonically increasing operation ID — if one rank calls
    a collective that another rank does not, all ranks will deadlock.  This means
    your driver script must be fully deterministic: avoid rank-conditional
    ``collect`` calls, early exits, or any branching that would cause different
    ranks to execute different query graphs.

    Must be invoked under the ``rrun`` launcher.  Use
    :func:`rapidsmpf.bootstrap.is_running_with_rrun` to test this at runtime.

    Parameters
    ----------
    executor_options
        Extra keyword arguments forwarded to the ``executor_options`` dict of
        :class:`~polars.lazyframe.engine_config.GPUEngine`.  The keys
        ``"runtime"``, ``"cluster"``, and ``"spmd"`` are reserved and may not
        be overridden.
    mr
        RMM device memory resource to use. Defaults to
        ``rmm.mr.CudaAsyncMemoryResource()`` when ``None``.
    options
        RapidsMPF options. Defaults to ``Options(get_environment_variables())``
        when ``None``.
    **engine_kwargs
        Extra keyword arguments forwarded directly to
        :class:`~polars.lazyframe.engine_config.GPUEngine`.  For example,
        pass ``parquet_options={"use_rapidsmpf_native": True}`` to enable
        native Parquet reads.

    Yields
    ------
    ctx : Context
        The active RapidsMPF context. Pass it to collective operations such as
        :func:`allgather_polars_dataframe`.
    engine : pl.GPUEngine
        A Polars GPU engine wired to ``ctx``. Pass it to
        ``LazyFrame.collect(engine=engine)`` on each rank.

    Raises
    ------
    RuntimeError
        If not running under the ``rrun`` launcher (i.e.
        :func:`rapidsmpf.bootstrap.is_running_with_rrun` returns ``False``).
        Launch with ``rrun -n <nproc> python -m pytest ...`` to fix this.
    ValueError
        If ``executor_options`` contains any of the reserved keys
        ``"runtime"``, ``"cluster"``, or ``"spmd"``.
    ValueError
        If ``engine_kwargs`` contains any of the reserved keys
        ``"raise_on_fail"``, ``"memory_resource"``, or ``"executor"``.

    Examples
    --------
    >>> with spmd_execution() as (ctx, engine):  # doctest: +SKIP
    ...     result = (
    ...         df.lazy().group_by("a").agg(pl.col("b").sum()).collect(engine=engine)
    ...     )
    ...     full = allgather_polars_dataframe(ctx=ctx, local_df=result, op_id=0)
    """
    if not bootstrap.is_running_with_rrun():
        raise RuntimeError(
            "spmd_execution() requires the rrun launcher. "
            "Use `rrun -n <nproc> python -m pytest ...` to run SPMD tests."
        )

    executor_options = executor_options or {}
    engine_kwargs = engine_kwargs or {}

    # Check for reserved keys.
    if bad := {"runtime", "cluster", "spmd"} & executor_options.keys():
        raise ValueError(f"executor_options may not contain reserved keys: {bad}")
    if bad := {"raise_on_fail", "memory_resource", "executor"} & engine_kwargs.keys():
        raise ValueError(f"engine_kwargs may not contain reserved keys: {bad}")

    options = options if options is not None else Options(get_environment_variables())
    mr = RmmResourceAdaptor(mr if mr is not None else rmm.mr.CudaAsyncMemoryResource())
    comm = bootstrap.create_ucxx_comm(
        progress_thread=ProgressThread(),
        type=bootstrap.BackendType.AUTO,
        options=options,
    )
    py_executor = ThreadPoolExecutor(
        max_workers=cast(
            int, executor_options.get("rapidsmpf_py_executor_max_workers", 1)
        ),
        thread_name_prefix="spmd-executor",
    )
    try:
        with Context.from_options(comm, mr, options) as ctx:
            engine = pl.GPUEngine(
                raise_on_fail=True,
                memory_resource=ctx.br().device_mr,
                executor="streaming",
                executor_options={
                    **executor_options,
                    "runtime": "rapidsmpf",
                    "cluster": "spmd",
                    "spmd": ContextSPMD(context=ctx, py_executor=py_executor),
                },
                **engine_kwargs,
            )
            yield ctx, engine
    finally:
        # The Context has already been exited above, so no work can be
        # pending in py_executor at this point; wait=False is safe.
        py_executor.shutdown(wait=False)
