# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""RapidsMPF streaming engine running on a Dask distributed cluster."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import os
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

import distributed
import ucxx._lib.libucxx as ucx_api
from rapidsmpf import bootstrap
from rapidsmpf.communicator.ucxx import barrier, get_root_ucxx_address, new_communicator
from rapidsmpf.config import (
    Options,
    get_environment_variables,
)
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.statistics import Statistics
from rapidsmpf.streaming.core.context import Context

import polars as pl

import rmm.mr

from cudf_polars.experimental.rapidsmpf.frontend.core import (
    StreamingEngine,
    check_reserved_keys,
    execute_ir_on_rank,
)
from cudf_polars.utils.config import DaskContext

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo, StatsCollector
    from cudf_polars.experimental.parallel import ConfigOptions
    from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
    from cudf_polars.utils.config import StreamingExecutor


@dataclasses.dataclass
class _WorkerContext:
    """Per-worker GPU resources stored on each Dask worker."""

    comm: Communicator | None
    ctx: Context | None
    py_executor: ThreadPoolExecutor | None
    mr: RmmResourceAdaptor | None
    statistics: Statistics | None


def _setup_root(
    nranks: int,
    rapidsmpf_options_as_bytes: bytes,
    *,
    uid: str,
    dask_worker: distributed.Worker | None = None,
) -> bytes:
    """
    Initialize the root rank on one Dask worker.

    Creates the UCXX communicator for rank 0 and stores partial state on the
    worker. The root UCXX address is returned so it can be forwarded to all
    other workers in phase 2.

    Parameters
    ----------
    nranks
        Total number of workers.
    rapidsmpf_options_as_bytes
        Serialized RapidsMPF options.
    uid
        Unique identifier for this cluster instance, used to namespace the
        per-worker attribute so multiple contexts can coexist on a worker.
    dask_worker
        Injected by ``distributed`` when called via :meth:`distributed.Client.run`.

    Returns
    -------
    Serialized root UCXX address for communicator bootstrap.
    """
    assert dask_worker is not None
    options = Options.deserialize(rapidsmpf_options_as_bytes)
    mr = RmmResourceAdaptor(rmm.mr.CudaAsyncMemoryResource())
    statistics = Statistics.from_options(mr, options)
    comm = new_communicator(
        nranks=nranks,
        ucx_worker=None,
        root_ucxx_address=None,
        options=options,
        progress_thread=ProgressThread(statistics),
    )
    setattr(
        dask_worker,
        f"_cudf_polars_mp_context_{uid}",
        _WorkerContext(
            comm=comm, ctx=None, py_executor=None, mr=mr, statistics=statistics
        ),
    )
    return get_root_ucxx_address(comm)


def _setup_worker(
    root_ucxx_address_as_bytes: bytes,
    nranks: int,
    rapidsmpf_options_as_bytes: bytes,
    executor_options: dict[str, object],
    *,
    uid: str,
    dask_worker: distributed.Worker | None = None,
) -> None:
    """
    Complete communicator bootstrap and create the streaming context.

    Must be called concurrently on all workers (including the root) so that
    the barrier can be reached by every rank simultaneously.

    Parameters
    ----------
    root_ucxx_address_as_bytes
        Serialized UCXX address returned by :func:`_setup_root`.
    nranks
        Total number of workers.
    rapidsmpf_options_as_bytes
        Serialized RapidsMPF options.
    executor_options
        Additional executor options.
    uid
        Unique identifier for this cluster instance, used to namespace the
        per-worker attribute so multiple contexts can coexist on a worker.
    dask_worker
        Injected by ``distributed`` when called via :meth:`distributed.Client.run`.
    """
    assert dask_worker is not None
    options = Options.deserialize(rapidsmpf_options_as_bytes)
    attr = f"_cudf_polars_mp_context_{uid}"
    mp_ctx: _WorkerContext | None = getattr(dask_worker, attr, None)

    if mp_ctx is None:
        # Non-root worker: create communicator now.
        mr = RmmResourceAdaptor(rmm.mr.CudaAsyncMemoryResource())
        statistics = Statistics.from_options(mr, options)
        root_addr = ucx_api.UCXAddress.create_from_buffer(root_ucxx_address_as_bytes)
        comm = new_communicator(
            nranks=nranks,
            ucx_worker=None,
            root_ucxx_address=root_addr,
            options=options,
            progress_thread=ProgressThread(statistics),
        )
    else:
        # Root worker: comm and mr were created in _setup_root.
        mr = mp_ctx.mr
        statistics = mp_ctx.statistics
        assert mp_ctx.comm is not None
        comm = mp_ctx.comm

    barrier(comm)
    ctx = Context.from_options(comm.logger, mr, options)
    # Set the current RMM device resource so all temporary allocations
    # in libcudf also use the same memory resource.
    rmm.mr.set_current_device_resource(ctx.br().device_mr)
    py_executor = ThreadPoolExecutor(
        max_workers=cast(
            int | None,
            executor_options.get("num_py_executors"),
        ),
        thread_name_prefix="dask-executor",
    )
    setattr(
        dask_worker,
        attr,
        _WorkerContext(
            comm=comm, ctx=ctx, py_executor=py_executor, mr=mr, statistics=statistics
        ),
    )


def _teardown_worker(
    *, uid: str, dask_worker: distributed.Worker | None = None
) -> None:
    """
    Release per-worker GPU resources.

    Shuts down the thread pool, drops the streaming context and communicator,
    and removes the worker attribute.

    Parameters
    ----------
    uid
        Unique identifier for the cluster instance to tear down.
    dask_worker
        Injected by ``distributed`` when called via :meth:`distributed.Client.run`.
    """
    assert dask_worker is not None
    attr = f"_cudf_polars_mp_context_{uid}"
    mp_ctx: _WorkerContext | None = getattr(dask_worker, attr, None)
    if mp_ctx is not None:
        if mp_ctx.py_executor is not None:
            mp_ctx.py_executor.shutdown(wait=True, cancel_futures=True)
        mp_ctx.ctx = None
        mp_ctx.comm = None
        mp_ctx.statistics = None
        mp_ctx.mr = None
        with contextlib.suppress(AttributeError):
            delattr(dask_worker, attr)


def _worker_evaluate(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions[StreamingExecutor],
    stats: StatsCollector,
    collective_id_map: dict[IR, list[int]],
    *,
    uid: str,
    collect_metadata: bool = False,
    dask_worker: distributed.Worker | None = None,
) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Execute a Polars IR query on this Dask worker's GPU.

    Parameters
    ----------
    ir
        Root IR node.
    partition_info
        Per-node partition metadata.
    config_options
        Executor configuration (``dask_context`` is already stripped).
    stats
        Statistics collector.
    collective_id_map
        Mapping from IR nodes to collective operation IDs.
    uid
        Unique identifier for the cluster instance, used to look up the
        per-worker context attribute.
    collect_metadata
        Whether to collect channel metadata.
    dask_worker
        Injected by ``distributed`` when called via :meth:`distributed.Client.run`.

    Returns
    -------
    result
        This worker's output fragment as a Polars DataFrame.
    metadata
        Collected channel metadata if ``collect_metadata`` is ``True``,
        otherwise ``None``.
    """
    assert dask_worker is not None
    mp_ctx: _WorkerContext = getattr(dask_worker, f"_cudf_polars_mp_context_{uid}")
    if mp_ctx.ctx is None or mp_ctx.comm is None or mp_ctx.py_executor is None:
        raise RuntimeError("_setup_worker must be called before _worker_evaluate")
    return execute_ir_on_rank(
        mp_ctx.ctx,
        mp_ctx.comm,
        mp_ctx.py_executor,
        ir,
        partition_info,
        config_options,
        stats,
        collective_id_map,
        collect_metadata=collect_metadata,
    )


def evaluate_pipeline_dask_mode(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions[StreamingExecutor],
    stats: StatsCollector,
    collective_id_map: dict[IR, list[int]],
    *,
    collect_metadata: bool = False,
    query_id: uuid.UUID,
) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Evaluate a RapidsMPF streaming pipeline in Dask mode.

    Dispatches :func:`_worker_evaluate` to every Dask worker via
    :meth:`distributed.Client.run`. Each worker executes the full pipeline
    on its local GPU and participates in collective operations through the
    shared UCXX communicator. Per-worker outputs are concatenated on the
    client before being returned.

    Parameters
    ----------
    ir
        The IR node.
    partition_info
        The partition information.
    config_options
        Executor configuration, including the ``dask_context`` handle.
    stats
        The statistics collector.
    collective_id_map
        Mapping from IR nodes to their pre-allocated collective operation IDs.
    collect_metadata
        Whether to collect runtime metadata.
    query_id
        A unique identifier for the query.

    Returns
    -------
    result
        Concatenated output from all Dask workers as a Polars DataFrame.
    metadata
        Collected channel metadata if ``collect_metadata`` is ``True``,
        otherwise ``None``.

    Raises
    ------
    RuntimeError
        If ``config_options.executor.dask_context`` is ``None``.
    """
    if config_options.executor.dask_context is None:
        raise RuntimeError("dask_context must be set when cluster='dask'")

    dask_context = config_options.executor.dask_context

    # Strip dask_context before pickling config_options for remote calls.
    worker_config = dataclasses.replace(
        config_options,
        executor=dataclasses.replace(config_options.executor, dask_context=None),
    )

    result_map = dask_context.client.run(
        functools.partial(_worker_evaluate, uid=dask_context.rapidsmpf_id),
        ir,
        partition_info,
        worker_config,
        stats,
        collective_id_map,
        collect_metadata=collect_metadata,
    )

    dfs: list[pl.DataFrame] = []
    metadata_collector: list[ChannelMetadata] = []
    for df, md in result_map.values():
        dfs.append(df)
        if md is not None:
            metadata_collector.extend(md)

    return pl.concat(dfs), metadata_collector or None


class DaskEngine(StreamingEngine):
    """
    Multi-GPU Polars engine for Dask distributed execution backed by RapidsMPF.

    Bootstraps a RapidsMPF UCXX cluster on top of a Dask distributed cluster
    and returns an engine that can be passed to ``LazyFrame.collect(engine=engine)``.

    If ``dask_client`` is provided, it is used directly and its lifetime is
    managed by the caller. If ``dask_client`` is ``None``, a
    :class:`dask_cuda.LocalCUDACluster` is created automatically (one worker
    per visible GPU) and torn down by :meth:`shutdown`.

    Prefer the context-manager form in scripts: it guarantees that workers are
    torn down even if an exception is raised. In interactive environments such
    as Jupyter notebooks, the direct form lets the cluster persist across
    multiple cells without tearing it down after every query.

    Parameters
    ----------
    dask_client
        An existing :class:`~distributed.Client` to use. If ``None``, a
        :class:`~dask_cuda.LocalCUDACluster` and a new client are created and
        owned by this engine.
    rapidsmpf_options
        RapidsMPF options forwarded to every worker. If ``None``, defaults to
        ``Options(get_environment_variables())``.
    executor_options
        Executor-specific options (e.g. ``max_rows_per_partition``).
    engine_options
        Engine-specific keyword arguments (e.g. ``raise_on_fail``,
        ``parquet_options``).

    Raises
    ------
    RuntimeError
        If called from within an ``rrun`` cluster.
    TypeError
        If ``executor_options`` or ``engine_options`` contains a reserved key.

    Examples
    --------
    Context-manager style:

    >>> with DaskEngine() as engine:  # doctest: +SKIP
    ...     result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)

    Bring-your-own client:

    >>> from distributed import Client
    >>> with Client("scheduler-address:8786") as dc:  # doctest: +SKIP
    ...     with DaskEngine(dask_client=dc) as engine:
    ...         result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)

    Jupyter / manual style:

    >>> engine = DaskEngine()  # doctest: +SKIP
    >>> result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)  # doctest: +SKIP
    >>> engine.shutdown()  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        dask_client: distributed.Client | None = None,
        rapidsmpf_options: Options | None = None,
        executor_options: dict[str, Any] | None = None,
        engine_options: dict[str, Any] | None = None,
    ) -> None:
        executor_options = executor_options or {}
        engine_options = engine_options or {}

        if bootstrap.is_running_with_rrun():
            raise RuntimeError(
                "DaskEngine must not be created from within an rrun cluster. Instead "
                "launch the rrun cluster separately and let this client connect to its "
                "cluster nodes."
            )

        check_reserved_keys(executor_options, engine_options)

        rapidsmpf_options = (
            rapidsmpf_options
            if rapidsmpf_options is not None
            else Options(get_environment_variables())
        )
        rapidsmpf_options.insert_if_absent({"num_streaming_threads": "4"})
        rapidsmpf_options_as_bytes = rapidsmpf_options.serialize()

        # Unique identifier for this cluster instance; namespaces the per-worker
        # attribute so multiple DaskEngine contexts can coexist on the same workers.
        uid = str(uuid.uuid4())

        owned_cluster: Any = None
        owned_client: distributed.Client | None = None
        if dask_client is None:
            import dask_cuda

            owned_cluster = dask_cuda.LocalCUDACluster()
            owned_client = distributed.Client(owned_cluster)
            dask_client = owned_client

        workers_info = dask_client.scheduler_info(n_workers=-1)["workers"]
        nranks = len(workers_info)
        root_worker = next(iter(workers_info))

        # Phase 1: initialize root communicator on one worker.
        root_result = dask_client.run(
            functools.partial(_setup_root, uid=uid),
            nranks,
            rapidsmpf_options_as_bytes,
            workers=[root_worker],
        )
        root_ucxx_address_as_bytes: bytes = root_result[root_worker]

        # Phase 2: complete bootstrap on all workers concurrently.
        # All workers call barrier() so they must all run simultaneously.
        dask_client.run(
            functools.partial(_setup_worker, uid=uid),
            root_ucxx_address_as_bytes,
            nranks,
            rapidsmpf_options_as_bytes,
            executor_options,
        )

        dask_ctx = DaskContext(
            client=dask_client,
            rapidsmpf_id=uid,
            owned_client=owned_client,
            owned_cluster=owned_cluster,
        )
        self._dask_context: DaskContext | None = dask_ctx
        super().__init__(
            nranks=nranks,
            executor_options={
                **executor_options,
                "runtime": "rapidsmpf",
                "cluster": "dask",
                "dask_context": dask_ctx,
            },
            engine_options={**engine_options, "memory_resource": None},
        )

    @property
    def _dask_ctx(self) -> DaskContext:
        if self._dask_context is None:
            raise RuntimeError("dask_context is not available after shutdown")
        return self._dask_context

    def gather_cluster_info(self) -> list[dict]:
        """
        Collect diagnostic information from every Dask worker.

        Returns
        -------
        List of info dicts, one per worker. Each dict contains
        ``pid``, ``hostname``, and ``cuda_visible_devices``.

        Examples
        --------
        >>> with DaskEngine() as engine:  # doctest: +SKIP
        ...     for info in engine.gather_cluster_info():
        ...         print(info)
        """

        def _get_info() -> dict:
            return {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }

        return list(self._dask_ctx.client.run(_get_info).values())

    def shutdown(self) -> None:
        """
        Shut down all Dask workers' GPU resources.

        If the cluster and client were created by this engine, they are also
        closed. Safe to call more than once. Must be called on the same thread
        that created the engine.

        Raises
        ------
        ExceptionGroup
            If one or more workers raise an unexpected exception during teardown.
        """
        if self._dask_context is None:
            return  # already shut down
        ctx = self._dask_context
        self._dask_context = None
        exceptions: list[Exception] = []
        try:
            ctx.client.run(functools.partial(_teardown_worker, uid=ctx.rapidsmpf_id))
        except Exception as e:
            exceptions.append(e)
        finally:
            if ctx.owned_client is not None:
                ctx.owned_client.close()
            if ctx.owned_cluster is not None:
                ctx.owned_cluster.close()
            super().shutdown()
        if exceptions:
            raise ExceptionGroup("Worker teardown failed", exceptions)

    @classmethod
    def from_options(
        cls,
        options: StreamingOptions,
        *,
        dask_client: distributed.Client | None = None,
    ) -> DaskEngine:
        """
        Create a :class:`DaskEngine` from a :class:`StreamingOptions` object.

        This is the recommended way to construct a ``DaskEngine`` for typical
        use. All RapidsMPF, executor, and engine options are read from
        ``options``; unset fields fall back to environment variables and then
        to built-in defaults.

        Parameters
        ----------
        options
            Unified streaming configuration.
        dask_client
            An existing :class:`distributed.Client` to use. If ``None``, a
            :class:`dask_cuda.LocalCUDACluster` is created automatically.

        Returns
        -------
        A new :class:`DaskEngine` instance.

        Examples
        --------
        >>> from cudf_polars.experimental.rapidsmpf.frontend.options import (
        ...     StreamingOptions,
        ... )
        >>> opts = StreamingOptions(num_streaming_threads=4, fallback_mode="silent")
        >>> with DaskEngine.from_options(opts) as engine:  # doctest: +SKIP
        ...     result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)
        """
        return cls(
            dask_client=dask_client,
            rapidsmpf_options=options.to_rapidsmpf_options(),
            executor_options=options.to_executor_options(),
            engine_options=options.to_engine_options(),
        )
