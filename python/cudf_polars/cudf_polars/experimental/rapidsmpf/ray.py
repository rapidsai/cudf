# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""RapidsMPF streaming engine running on a Ray cluster."""

from __future__ import annotations

import dataclasses
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import ray
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
from rapidsmpf.streaming.core.actor import run_actor_network
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import polars as pl

import rmm.mr

from cudf_polars.containers import DataFrame
from cudf_polars.dsl.ir import IRExecutionContext
from cudf_polars.experimental.rapidsmpf.core import generate_network
from cudf_polars.experimental.rapidsmpf.utils import empty_table_chunk
from cudf_polars.experimental.utils import _concat

if TYPE_CHECKING:
    from collections.abc import Iterator, MutableMapping

    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.cudf.channel_metadata import ChannelMetadata

    from cudf_polars.dsl.ir import IR
    from cudf_polars.experimental.base import PartitionInfo, StatsCollector
    from cudf_polars.experimental.parallel import ConfigOptions
    from cudf_polars.utils.config import StreamingExecutor


def evaluate_pipeline_ray_mode(
    ir: IR,
    partition_info: MutableMapping[IR, PartitionInfo],
    config_options: ConfigOptions[StreamingExecutor],
    stats: StatsCollector,
    collective_id_map: dict[IR, list[int]],
    *,
    collect_metadata: bool = False,
) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
    """
    Evaluate a RapidsMPF streaming pipeline in Ray mode.

    The query is dispatched concurrently to every :class:`RankActor` in the
    Ray cluster. Each actor evaluates the full pipeline on its local GPU and
    participates in collective operations through the shared UCXX
    communicator. The per-rank outputs are concatenated on the client before
    being returned.

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
    result
        Concatenated output from all Ray actors as a Polars DataFrame.
    metadata
        Collected channel metadata if ``collect_metadata`` is ``True``,
        otherwise ``None``.

    Raises
    ------
    RuntimeError
        If the configured executor runtime is not ``"rapidsmpf"``.
    RuntimeError
        If ``config_options.executor.ray_client`` is ``None``.
    """
    if config_options.executor.runtime != "rapidsmpf":
        raise RuntimeError("Runtime must be rapidsmpf")
    if config_options.executor.ray_client is None:
        raise RuntimeError("ray_client must be set when cluster='ray'")
    rank_actors = config_options.executor.ray_client._rank_actors

    # Strip ray_client before pickling config_options for remote calls:
    # actors don't need the full actor list, and sending actor handles to each
    # actor is wasteful.
    actor_config_options = dataclasses.replace(
        config_options,
        executor=dataclasses.replace(config_options.executor, ray_client=None),
    )

    # ir, partition_info, stats, and collective_id_map must be pickled together
    # so that the IR-node keys in partition_info / collective_id_map are the
    # same objects as the nodes in the ir tree after deserialization.
    query_bundle = (ir, partition_info, stats, collective_id_map)
    result = ray.get(
        [
            rank.evaluate_polars_ir.remote(
                query_bundle,
                actor_config_options,
                collect_metadata=collect_metadata,
            )
            for rank in rank_actors
        ]
    )
    dfs: list[pl.DataFrame] = []
    metadata_collector: list[ChannelMetadata] = []
    for df, md in result:
        dfs.append(df)
        if md is not None:
            metadata_collector.extend(md)

    return pl.concat(dfs), metadata_collector or None


@ray.remote(
    max_restarts=0,
    max_task_retries=0,
    num_cpus=0,
    num_gpus=1,
)
class RankActor:
    """
    Ray actor that owns one GPU and participates in a RapidsMPF cluster.

    Each actor manages its own memory resource, statistics collector,
    communicator, and streaming context. Collectively, the actors form a
    SPMD execution cluster used by the client-side Ray integration.

    Parameters
    ----------
    nranks
        Total number of actors, typically one per GPU in the Ray cluster.
    rapidsmpf_options_as_bytes
        Serialized RapidsMPF options produced by
        :meth:`rapidsmpf.config.Options.serialize`.
    executor_options
        Additional executor options forwarded from the client.
    """

    def __init__(
        self,
        *,
        nranks: int,
        rapidsmpf_options_as_bytes: bytes,
        executor_options: dict[str, object],
    ) -> None:
        self._mr = RmmResourceAdaptor(rmm.mr.CudaAsyncMemoryResource())
        self._rapidsmpf_options: Options = Options.deserialize(
            rapidsmpf_options_as_bytes
        )
        self._statistics: Statistics = Statistics.from_options(
            self._mr, self._rapidsmpf_options
        )
        self._nranks: int = nranks
        self._py_executor = ThreadPoolExecutor(
            max_workers=cast(
                int, executor_options.get("rapidsmpf_py_executor_max_workers", 1)
            ),
            thread_name_prefix="ray-executor",
        )
        self._comm: Communicator | None = None
        self._ctx: Context | None = None

    def setup_root(self) -> bytes:
        """
        Initialize this actor as the root rank.

        The root actor creates a new UCXX communicator without an existing
        root address. The resulting root address is returned so it can be
        distributed to all other actors during bootstrap.

        Returns
        -------
        Serialized UCXX root address for communicator bootstrap.
        """
        self._comm = new_communicator(
            nranks=self._nranks,
            ucx_worker=None,
            root_ucxx_address=None,
            options=self._rapidsmpf_options,
            progress_thread=ProgressThread(self._statistics),
        )
        return get_root_ucxx_address(self._comm)

    def setup_worker(self, root_ucxx_address_as_bytes: bytes) -> None:
        """
        Complete communicator bootstrap and create the streaming context.

        This method must be called concurrently on all actors, including the
        root. Non-root actors connect to the root using the provided UCXX
        address. Once all ranks have joined, the actors synchronize with a
        barrier and create their RapidsMPF streaming contexts.

        Parameters
        ----------
        root_ucxx_address_as_bytes
            Serialized UCXX root address returned by :meth:`setup_root`.
        """
        if self._comm is None:
            root_ucxx_address = ucx_api.UCXAddress.create_from_buffer(
                root_ucxx_address_as_bytes
            )
            self._comm = new_communicator(
                nranks=self._nranks,
                ucx_worker=None,
                root_ucxx_address=root_ucxx_address,
                options=self._rapidsmpf_options,
                progress_thread=ProgressThread(self._statistics),
            )
        barrier(self._comm)
        self._ctx = Context.from_options(
            self._comm.logger, self._mr, self._rapidsmpf_options
        )
        # Set the current RMM device resource so all temporary allocations
        # in libcudf also use the same memory resource.
        rmm.mr.set_current_device_resource(self._ctx.br().device_mr)

    def shutdown(self) -> None:
        """
        Release actor-owned resources and exit the process.

        This shuts down the local Python executor, drops communicator and
        memory-resource references, and then terminates the Ray actor process.
        """
        self._py_executor.shutdown(wait=True, cancel_futures=True)
        self._comm = None
        self._mr = None
        ray.actor.exit_actor()

    def get_info(self) -> dict:
        """
        Return diagnostic information about actor placement.

        Returns
        -------
        Diagnostic information about this actor's placement and state.
        """
        return {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "node_id": ray.get_runtime_context().get_node_id(),
        }

    def evaluate_polars_ir(
        self,
        query_bundle: tuple[
            IR,
            MutableMapping[IR, PartitionInfo],
            StatsCollector,
            dict[IR, list[int]],
        ],
        config_options: ConfigOptions[StreamingExecutor],
        *,
        collect_metadata: bool,
    ) -> tuple[pl.DataFrame, list[ChannelMetadata] | None]:
        """
        Execute a Polars IR query on this actor's GPU.

        The IR is lowered to a RapidsMPF actor network, executed locally, and
        the resulting output messages are assembled into a Polars DataFrame.
        Collective operations in the network communicate with peer actors
        through the shared UCXX communicator.

        Parameters
        ----------
        query_bundle
            Tuple of ``(ir, partition_info, stats, collective_id_map)``.
            Bundled into a single argument so that all four objects are
            pickled together, preserving object identity between IR-node
            keys in ``partition_info`` / ``collective_id_map`` and the
            nodes in the ``ir`` tree.
        config_options
            Executor configuration forwarded from the client.
        collect_metadata
            If ``True``, collect channel metadata during execution.

        Returns
        -------
        result
            This rank's output fragment as a Polars DataFrame.
        metadata
            Collected channel metadata if ``collect_metadata`` is ``True``,
            otherwise ``None``.

        Raises
        ------
        AssertionError
            If :meth:`setup_worker` has not been called first.
        """
        ir, partition_info, stats, collective_id_map = query_bundle
        assert self._ctx is not None, (
            "setup_worker must be called before evaluate_polars_ir"
        )
        ir_context = IRExecutionContext(get_cuda_stream=self._ctx.get_stream_from_pool)
        metadata_collector: list[ChannelMetadata] | None = (
            [] if collect_metadata else None
        )

        nodes, output = generate_network(
            self._ctx,
            self._comm,
            ir,
            partition_info,
            config_options,
            stats,
            ir_context=ir_context,
            collective_id_map=collective_id_map,
            metadata_collector=metadata_collector,
        )

        run_actor_network(actors=nodes, py_executor=self._py_executor)

        messages = output.release()
        chunks = [
            TableChunk.from_message(msg).make_available_and_spill(
                self._ctx.br(), allow_overbooking=True
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
            # No chunks received, create an empty DataFrame with the correct schema.
            stream = ir_context.get_cuda_stream()
            chunk = empty_table_chunk(ir, self._ctx, stream)
            df = DataFrame.from_table(
                chunk.table_view(),
                list(ir.schema.keys()),
                list(ir.schema.values()),
                stream,
            )
        result = df.to_polars()
        return result, metadata_collector


class RayClient:
    """Client-side handle for the distributed cudf-polars execution."""

    def __init__(self, rank_actors: list[Any]) -> None:
        self._rank_actors: list[Any] = rank_actors

    @property
    def nranks(self) -> int:
        """
        Number of Ray rank actors.

        Returns
        -------
        Number of ranks/nodes in the Ray cluster.
        """
        return len(self._rank_actors)

    def gather_cluster_info(self) -> list[dict]:
        """
        Collect diagnostic information from every rank actor.

        Returns
        -------
        List of info dicts (see :meth:`RankActor.get_info`), one per rank
        in rank order.

        Examples
        --------
        >>> with ray_execution() as (ray_client, engine):  # doctest: +SKIP
        ...     for i, info in enumerate(ray_client.gather_cluster_info()):
        ...         print(f"rank {i}: {info}")
        """
        return ray.get([rank.get_info.remote() for rank in self._rank_actors])


@contextmanager
def ray_execution(
    *,
    rapidsmpf_options: Options | None = None,
    executor_options: dict[str, object] | None = None,
    engine_kwargs: dict[str, Any] | None = None,
    ray_init_kwargs: dict[str, object] | None = None,
) -> Iterator[tuple[RayClient, pl.GPUEngine]]:
    """
    Create a RapidsMPF Ray cluster and matching Polars GPU engine.

    If Ray is not already initialized, this context manager calls
    :func:`ray.init` on entry and :func:`ray.shutdown` on exit. If Ray is
    already initialized, cluster lifetime remains managed by the caller.

    Parameters
    ----------
    rapidsmpf_options
        RapidsMPF options forwarded to every actor. If ``None``, defaults to
        ``Options(get_environment_variables())``.
    executor_options
        Additional key-value pairs forwarded to the Polars executor options.
    engine_kwargs
        Additional keyword arguments forwarded to :class:`polars.GPUEngine`.
    ray_init_kwargs
        Keyword arguments forwarded to :func:`ray.init` when Ray is not
        already initialized.

    Yields
    ------
    ray_client
        Client-side handle to the Ray actor cluster.
    engine
        Polars GPU engine configured to execute through RapidsMPF on Ray.

    Raises
    ------
    RuntimeError
        If called from within an ``rrun`` cluster.
    RuntimeError
        If not all GPUs in the Ray cluster are free at startup.
    RuntimeError
        If no GPUs are available in the Ray cluster.
    ValueError
        If ``executor_options`` contains a reserved key.
    ValueError
        If ``engine_kwargs`` contains a reserved key.

    Examples
    --------
    >>> with ray_execution() as (ray_client, engine):  # doctest: +SKIP
    ...     result = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=engine)
    """
    executor_options = executor_options or {}
    engine_kwargs = engine_kwargs or {}
    ray_init_kwargs = ray_init_kwargs or {}

    if bootstrap.is_running_with_rrun():
        raise RuntimeError(
            "ray_execution() must not be called from within an rrun cluster. Instead "
            "launch the rrun cluster separately and let this client connect to its "
            "cluster nodes."
        )

    # Check for reserved keys.
    if bad := {"runtime", "cluster", "spmd", "ray_client"} & executor_options.keys():
        raise ValueError(f"executor_options may not contain reserved keys: {bad}")
    if bad := {"memory_resource", "executor"} & engine_kwargs.keys():
        raise ValueError(f"engine_kwargs may not contain reserved keys: {bad}")

    rapidsmpf_options = (
        rapidsmpf_options
        if rapidsmpf_options is not None
        else Options(get_environment_variables())
    )
    rapidsmpf_options_as_bytes = rapidsmpf_options.serialize()

    ray_was_initialized: bool = ray.is_initialized()
    if not ray_was_initialized:
        # Prevent Ray from overriding CUDA_VISIBLE_DEVICES to "" when a worker
        # process starts with zero visible GPUs (e.g. the driver process itself).
        # Without this, Ray's accelerator detection resets the variable before our
        # actors acquire their GPU assignment, hiding all GPUs from CUDA.
        os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
        ray.init(**ray_init_kwargs)

    total_gpus = int(ray.cluster_resources().get("GPU", 0.0))
    free_gpus = int(ray.available_resources().get("GPU", 0.0))
    if total_gpus != free_gpus:
        raise RuntimeError(
            "Ray execution expects all GPUs in the Ray cluster to be available at startup"
        )
    if free_gpus == 0:
        raise RuntimeError("No available GPUs in the Ray cluster at startup")

    # Create one actor per GPU. Ray adds .remote() dynamically; no type stubs.
    rank_actors: list[Any] = [
        RankActor.remote(  # type: ignore[attr-defined]
            nranks=free_gpus,
            executor_options=executor_options,
            rapidsmpf_options_as_bytes=rapidsmpf_options_as_bytes,
        )
        for _ in range(free_gpus)
    ]

    root_ucxx_address_as_bytes = ray.get(rank_actors[0].setup_root.remote())
    # Call setup_worker on all actors concurrently, including the root.
    # The root skips communicator creation and proceeds directly to the barrier.
    # Non-root actors create their communicators and then join the barrier.
    ray.get(
        [rank.setup_worker.remote(root_ucxx_address_as_bytes) for rank in rank_actors]
    )

    try:
        ray_client = RayClient(rank_actors)
        engine = pl.GPUEngine(
            memory_resource=None,
            executor="streaming",
            executor_options={
                **executor_options,
                "runtime": "rapidsmpf",
                "cluster": "ray",
                "ray_client": ray_client,
            },
            **engine_kwargs,
        )
        yield ray_client, engine
    finally:
        for a in rank_actors:
            try:
                ray.get(a.shutdown.remote())
            except ray.exceptions.RayActorError:
                pass  # expected: exit_actor() terminates the process immediately.
            except Exception as e:
                print(f"shutdown error: {e}")
        if not ray_was_initialized:
            ray.shutdown()
