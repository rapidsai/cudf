# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions/classes for running the PDS-H and PDS-DS benchmarks."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import io
import itertools
import json
import logging
import os
import statistics
import sys
import textwrap
import time
import traceback
import uuid
import warnings
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, assert_never

import nvtx

import polars as pl

import rmm.statistics

try:
    import duckdb

    duckdb_err = None
except ImportError as e:
    duckdb = None
    duckdb_err = e

try:
    import pynvml
except ImportError:
    pynvml = None

try:
    from cudf_polars.dsl.ir import IRExecutionContext
    from cudf_polars.dsl.tracing import Scope
    from cudf_polars.dsl.translate import Translator
    from cudf_polars.experimental.explain import explain_query
    from cudf_polars.experimental.parallel import evaluate_streaming
    from cudf_polars.testing.asserts import assert_gpu_result_equal
    from cudf_polars.utils.config import ConfigOptions

    CUDF_POLARS_AVAILABLE = True
except ImportError:
    CUDF_POLARS_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cudf_polars.experimental.explain import SerializablePlan


try:
    import structlog
    import structlog.contextvars
    import structlog.processors
    import structlog.stdlib
except ImportError:
    _HAS_STRUCTLOG = False
else:
    _HAS_STRUCTLOG = True


ExecutorType = Literal["in-memory", "streaming", "cpu"]


@dataclasses.dataclass
class Record:
    """Results for a single run of a single PDS-H query."""

    query: int
    iteration: int
    duration: float
    shuffle_stats: dict[str, dict[str, int | float]] | None = None
    traces: list[dict[str, Any]] | None = None

    @classmethod
    def new(
        cls,
        query: int,
        iteration: int,
        duration: float,
        shuffle_stats: dict[str, dict[str, int | float]] | None = None,
        traces: list[dict[str, Any]] | None = None,
    ) -> Record:
        """Create a Record from plain data."""
        return cls(
            query=query,
            iteration=iteration,
            duration=duration,
            shuffle_stats=shuffle_stats,
            traces=traces,
        )


@dataclasses.dataclass
class VersionInfo:
    """Information about the commit of the software used to run the query."""

    version: str
    commit: str


@dataclasses.dataclass
class PackageVersions:
    """Information about the versions of the software used to run the query."""

    cudf_polars: str | VersionInfo
    polars: str
    python: str
    rapidsmpf: str | VersionInfo | None

    @classmethod
    def collect(cls) -> PackageVersions:
        """Collect the versions of the software used to run the query."""
        packages = [
            "cudf_polars",
            "polars",
            "rapidsmpf",
        ]
        versions: dict[str, str | VersionInfo | None] = {}
        for name in packages:
            try:
                package = importlib.import_module(name)
            except (AttributeError, ImportError):
                versions[name] = None
            else:
                if name in ("cudf_polars", "rapidsmpf"):
                    versions[name] = VersionInfo(
                        version=package.__version__,
                        commit=package.__git_commit__,
                    )
                else:
                    versions[name] = package.__version__

        versions["python"] = ".".join(str(v) for v in sys.version_info[:3])
        # we manually ensure that only cudf-polars and rapidsmpf have a VersionInfo
        return cls(**versions)  # type: ignore[arg-type]


@dataclasses.dataclass
class GPUInfo:
    """Information about a specific GPU."""

    name: str
    index: int
    free_memory: int | None
    used_memory: int | None
    total_memory: int | None

    @classmethod
    def from_index(cls, index: int) -> GPUInfo:
        """Create a GPUInfo from an index."""
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        try:
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return cls(
                name=pynvml.nvmlDeviceGetName(handle),
                index=index,
                free_memory=memory.free,
                used_memory=memory.used,
                total_memory=memory.total,
            )
        except pynvml.NVMLError_NotSupported:
            # Happens on systems without traditional GPU memory (e.g., Grace Hopper),
            # where nvmlDeviceGetMemoryInfo is not supported.
            # See: https://github.com/rapidsai/cudf/issues/19427
            return cls(
                name=pynvml.nvmlDeviceGetName(handle),
                index=index,
                free_memory=None,
                used_memory=None,
                total_memory=None,
            )


@dataclasses.dataclass
class HardwareInfo:
    """Information about the hardware used to run the query."""

    gpus: list[GPUInfo]
    # TODO: ucx

    @classmethod
    def collect(cls) -> HardwareInfo:
        """Collect the hardware information."""
        if pynvml is not None:
            pynvml.nvmlInit()
            gpus = [GPUInfo.from_index(i) for i in range(pynvml.nvmlDeviceGetCount())]
        else:
            # No GPUs -- probably running in CPU mode
            gpus = []
        return cls(gpus=gpus)


def _infer_scale_factor(name: str, path: str | Path, suffix: str) -> int | float:
    if "pdsh" in name:
        supplier = get_data(path, "supplier", suffix)
        num_rows = supplier.select(pl.len()).collect().item(0, 0)
        return num_rows / 10_000

    elif "pdsds" in name:
        # TODO: Keep a map of SF-row_count because of nonlinear scaling
        # See: https://www.tpc.org/TPC_Documents_Current_Versions/pdf/TPC-DS_v4.0.0.pdf pg.46
        customer = get_data(path, "promotion", suffix)
        num_rows = customer.select(pl.len()).collect().item(0, 0)
        return num_rows / 300

    else:
        raise ValueError(f"Invalid benchmark script name: '{name}'.")


@dataclasses.dataclass(kw_only=True)
class RunConfig:
    """Results for a PDS-H or PDS-DS query run."""

    queries: list[int]
    suffix: str
    executor: ExecutorType
    runtime: str
    stream_policy: str | None
    cluster: str
    scheduler: str  # Deprecated, kept for backward compatibility
    n_workers: int
    versions: PackageVersions = dataclasses.field(
        default_factory=PackageVersions.collect
    )
    records: dict[int, list[Record]] = dataclasses.field(default_factory=dict)
    plans: dict[int, SerializablePlan] = dataclasses.field(default_factory=dict)
    dataset_path: Path
    scale_factor: int | float
    shuffle: Literal["rapidsmpf", "tasks"] | None = None
    gather_shuffle_stats: bool = False
    broadcast_join_limit: int | None = None
    blocksize: int | None = None
    max_rows_per_partition: int | None = None
    threads: int
    iterations: int
    timestamp: str = dataclasses.field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    hardware: HardwareInfo = dataclasses.field(default_factory=HardwareInfo.collect)
    run_id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    rmm_async: bool
    rapidsmpf_oom_protection: bool
    rapidsmpf_spill: bool
    spill_device: float
    query_set: str
    collect_traces: bool = False
    stats_planning: bool
    dynamic_planning: bool | None = None
    max_io_threads: int
    native_parquet: bool
    spill_to_pinned_memory: bool
    extra_info: dict[str, Any] = dataclasses.field(default_factory=dict)
    fallback_mode: str | None = None

    def __post_init__(self) -> None:  # noqa: D105
        if self.gather_shuffle_stats and self.shuffle != "rapidsmpf":
            raise ValueError(
                "gather_shuffle_stats is only supported when shuffle='rapidsmpf'."
            )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RunConfig:
        """Create a RunConfig from command line arguments."""
        executor: ExecutorType = args.executor
        cluster = args.cluster
        scheduler = args.scheduler
        runtime = args.runtime
        stream_policy = args.stream_policy

        # Handle "auto" stream policy
        if stream_policy == "auto":
            stream_policy = None

        # Deal with deprecated scheduler argument
        # and non-streaming executors
        if executor == "in-memory" or executor == "cpu":
            cluster = None
            scheduler = None
        elif scheduler is not None:
            if cluster is not None:
                raise ValueError(
                    "Cannot specify both -s/--scheduler and -c/--cluster. "
                    "Please use -c/--cluster only."
                )
            else:
                warnings.warn(
                    "The -s/--scheduler argument is deprecated. Use -c/--cluster instead.",
                    FutureWarning,
                    stacklevel=2,
                )
            cluster = "single" if scheduler == "synchronous" else "distributed"
        elif cluster is not None:
            scheduler = "synchronous" if cluster == "single" else "distributed"
        else:
            cluster = "single"
            scheduler = "synchronous"

        path = args.path
        name = args.query_set
        scale_factor = args.scale

        if scale_factor is None:
            if "pdsds" in name:
                raise ValueError(
                    "--scale is required for PDS-DS benchmarks.\n"
                    "TODO: This will be inferred once we maintain a map of scale factors to row counts."
                )
            if path is None:
                raise ValueError(
                    "Must specify --root and --scale if --path is not specified."
                )
            # For PDS-H, infer scale factor based on row count
            scale_factor = _infer_scale_factor(name, path, args.suffix)
        if path is None:
            path = f"{args.root}/scale-{scale_factor}"

        scale_factor = float(scale_factor)
        try:
            scale_factor_int = int(scale_factor)
        except ValueError:
            pass
        else:
            if scale_factor_int == scale_factor:
                scale_factor = scale_factor_int

        skip_scale_factor_inference = (
            "LIBCUDF_IO_REROUTE_LOCAL_DIR_PATTERN" in os.environ
        ) and ("LIBCUDF_IO_REROUTE_REMOTE_DIR_PATTERN" in os.environ)

        if (
            "pdsh" in name
            and args.scale is not None
            and skip_scale_factor_inference is False
        ):
            # Validate the user-supplied scale factor
            sf_inf = _infer_scale_factor(name, path, args.suffix)
            rel_error = abs((scale_factor - sf_inf) / sf_inf)
            if rel_error > 0.01:
                raise ValueError(
                    f"Specified scale factor is {args.scale}, "
                    f"but the inferred scale factor is {sf_inf}."
                )

        return cls(
            queries=args.query,
            executor=executor,
            cluster=cluster,
            scheduler=scheduler,
            runtime=runtime,
            stream_policy=stream_policy,
            n_workers=args.n_workers,
            shuffle=args.shuffle,
            gather_shuffle_stats=args.rapidsmpf_dask_statistics,
            broadcast_join_limit=args.broadcast_join_limit,
            dataset_path=path,
            scale_factor=scale_factor,
            blocksize=args.blocksize,
            threads=args.threads,
            iterations=args.iterations,
            suffix=args.suffix,
            rmm_async=args.rmm_async,
            rapidsmpf_oom_protection=args.rapidsmpf_oom_protection,
            spill_device=args.spill_device,
            rapidsmpf_spill=args.rapidsmpf_spill,
            max_rows_per_partition=args.max_rows_per_partition,
            query_set=args.query_set,
            collect_traces=args.collect_traces,
            stats_planning=args.stats_planning,
            dynamic_planning=args.dynamic_planning,
            max_io_threads=args.max_io_threads,
            native_parquet=args.native_parquet,
            extra_info=args.extra_info,
            spill_to_pinned_memory=args.spill_to_pinned_memory,
            fallback_mode=args.fallback_mode,
        )

    def serialize(self, engine: pl.GPUEngine | None) -> dict:
        """Serialize the run config to a dictionary."""
        result = dataclasses.asdict(self)
        result["run_id"] = str(self.run_id)

        if engine is not None:
            config_options = ConfigOptions.from_polars_engine(engine)
            result["config_options"] = dataclasses.asdict(config_options)
        return result

    def summarize(self) -> None:
        """Print a summary of the results."""
        print("Iteration Summary")
        print("=======================================")

        for query, records in self.records.items():
            print(f"query: {query}")
            print(f"path: {self.dataset_path}")
            print(f"scale_factor: {self.scale_factor}")
            print(f"executor: {self.executor}")
            print(f"stream_policy: {self.stream_policy}")
            if self.executor == "streaming":
                print(f"runtime: {self.runtime}")
                print(f"cluster: {self.cluster}")
                print(f"blocksize: {self.blocksize}")
                print(f"shuffle_method: {self.shuffle}")
                print(f"broadcast_join_limit: {self.broadcast_join_limit}")
                print(f"stats_planning: {self.stats_planning}")
                if self.runtime == "rapidsmpf":
                    print(f"native_parquet: {self.native_parquet}")
                    print(f"dynamic_planning: {self.dynamic_planning}")
                if self.cluster == "distributed":
                    print(f"n_workers: {self.n_workers}")
                    print(f"threads: {self.threads}")
                    print(f"rmm_async: {self.rmm_async}")
                    print(f"rapidsmpf_oom_protection: {self.rapidsmpf_oom_protection}")
                    print(f"spill_device: {self.spill_device}")
                    print(f"rapidsmpf_spill: {self.rapidsmpf_spill}")
            if len(records) > 0:
                print(f"iterations: {self.iterations}")
                print("---------------------------------------")
                print(f"min time : {min(record.duration for record in records):0.4f}")
                print(f"max time : {max(record.duration for record in records):0.4f}")
                print(
                    f"mean time: {statistics.mean(record.duration for record in records):0.4f}"
                )
                print("=======================================")
        total_mean_time = sum(
            statistics.mean(record.duration for record in records)
            for records in self.records.values()
            if records
        )
        print(f"Total mean time across all queries: {total_mean_time:.4f} seconds")


def get_data(path: str | Path, table_name: str, suffix: str = "") -> pl.LazyFrame:
    """Get table from dataset."""
    return pl.scan_parquet(f"{path}/{table_name}{suffix}")


def get_executor_options(
    run_config: RunConfig, benchmark: Any = None
) -> dict[str, Any]:
    """Generate executor_options for GPUEngine."""
    executor_options: dict[str, Any] = {}

    if run_config.executor == "streaming":
        if run_config.blocksize:
            executor_options["target_partition_size"] = run_config.blocksize
        if run_config.max_rows_per_partition:
            executor_options["max_rows_per_partition"] = (
                run_config.max_rows_per_partition
            )
        if run_config.shuffle:
            executor_options["shuffle_method"] = run_config.shuffle
        if run_config.broadcast_join_limit:
            executor_options["broadcast_join_limit"] = run_config.broadcast_join_limit
        if run_config.rapidsmpf_spill:
            executor_options["rapidsmpf_spill"] = run_config.rapidsmpf_spill
        if run_config.fallback_mode:
            executor_options["fallback_mode"] = run_config.fallback_mode
        if run_config.cluster == "distributed":
            executor_options["cluster"] = "distributed"
        executor_options["stats_planning"] = {
            "use_reduction_planning": run_config.stats_planning,
            "use_sampling": (
                # Always allow row-group sampling for rapidsmpf runtime
                run_config.stats_planning or run_config.runtime == "rapidsmpf"
            ),
        }
        executor_options["client_device_threshold"] = run_config.spill_device
        executor_options["runtime"] = run_config.runtime
        executor_options["max_io_threads"] = run_config.max_io_threads
        executor_options["spill_to_pinned_memory"] = run_config.spill_to_pinned_memory
        if run_config.dynamic_planning:
            # Pass empty dict to enable with defaults; None means disabled
            executor_options["dynamic_planning"] = {}

    if (
        benchmark
        and benchmark.__name__ == "PDSHQueries"
        and run_config.executor == "streaming"
        # Only use the unique_fraction config if stats_planning is disabled
        and not run_config.stats_planning
        and not run_config.dynamic_planning
    ):
        executor_options["unique_fraction"] = {
            "c_custkey": 0.05,
            "l_orderkey": 1.0,
            "l_partkey": 0.1,
            "o_custkey": 0.25,
        }

    return executor_options


def print_query_plan(
    q_id: int,
    q: pl.LazyFrame,
    args: argparse.Namespace,
    run_config: RunConfig,
    engine: None | pl.GPUEngine = None,
    *,
    print_plans: bool = True,
) -> tuple[str | None, str | None]:
    """Print the query plan."""
    logical_plan = plan = None
    if run_config.executor == "cpu":
        if args.explain_logical:
            logical_plan = q.explain()
        if args.explain:
            plan = q.show_graph(engine="streaming", plan_stage="physical")
    elif CUDF_POLARS_AVAILABLE:
        assert isinstance(engine, pl.GPUEngine)
        if args.explain_logical:
            logical_plan = explain_query(q, engine, physical=False)
        if args.explain and run_config.executor == "streaming":
            plan = explain_query(q, engine)
    else:
        raise RuntimeError(
            "Cannot provide the logical or physical plan because cudf_polars is not installed."
        )

    if print_plans:
        if logical_plan:
            print(f"\nQuery {q_id} - Logical plan\n")
            print(logical_plan)
        if plan:
            print(f"\nQuery {q_id} - Physical plan\n")
            print(plan)

    return logical_plan, plan


def initialize_dask_cluster(run_config: RunConfig, args: argparse.Namespace):  # type: ignore[no-untyped-def]
    """
    Initialize a Dask distributed cluster.

    This function either creates a new LocalCUDACluster or connects to an
    existing Dask cluster depending on the provided arguments.

    Parameters
    ----------
    run_config : RunConfig
        The run configuration.
    args : argparse.Namespace
        Parsed command line arguments. If ``args.scheduler_address`` or
        ``args.scheduler_file`` is provided, we connect to an existing
        cluster instead of creating a LocalCUDACluster.

    Returns
    -------
    Client or None
        A Dask distributed Client, or None if not using distributed mode.
    """
    if run_config.cluster != "distributed":
        return None

    from distributed import Client

    # Check if we should connect to an existing cluster
    scheduler_address = args.scheduler_address
    scheduler_file = args.scheduler_file

    if scheduler_address is not None:
        # Connect to existing cluster via scheduler address
        client = Client(address=scheduler_address)
        n_workers = client.scheduler_info()["n_workers"]
        print(
            f"Connected to existing Dask cluster at {scheduler_address} "
            f"with {n_workers} workers"
        )
    elif scheduler_file is not None:
        # Connect to existing cluster via scheduler file
        client = Client(scheduler_file=scheduler_file)
        n_workers = client.scheduler_info()["n_workers"]
        print(
            f"Connected to existing Dask cluster via scheduler file: {scheduler_file} "
            f"with {n_workers} workers"
        )
    else:
        # Create a new LocalCUDACluster
        from dask_cuda import LocalCUDACluster

        kwargs = {
            "n_workers": run_config.n_workers,
            "dashboard_address": ":8585",
            "protocol": args.protocol,
            "rmm_pool_size": args.rmm_pool_size,
            "rmm_async": args.rmm_async,
            "rmm_release_threshold": args.rmm_release_threshold,
            "threads_per_worker": run_config.threads,
        }

        client = Client(LocalCUDACluster(**kwargs))
        client.wait_for_workers(run_config.n_workers)

    if run_config.shuffle != "tasks":
        try:
            from rapidsmpf.config import Options
            from rapidsmpf.integrations.dask import bootstrap_dask_cluster

            bootstrap_dask_cluster(
                client,
                options=Options(
                    {
                        "dask_spill_device": str(run_config.spill_device),
                        "dask_spill_to_pinned_memory": str(
                            run_config.spill_to_pinned_memory
                        ),
                        "dask_statistics": str(args.rapidsmpf_dask_statistics),
                        "dask_print_statistics": str(args.rapidsmpf_print_statistics),
                        "dask_oom_protection": str(args.rapidsmpf_oom_protection),
                    }
                ),
            )
            # Setting this globally makes the peak statistics not meaningful
            # across queries / iterations. But doing it per query isn't worth
            # the effort right now.
            client.run(rmm.statistics.enable_statistics)
        except ImportError as err:
            if run_config.shuffle == "rapidsmpf":
                raise ImportError(
                    "rapidsmpf is required for shuffle='rapidsmpf' but is not installed."
                ) from err

    return client


def execute_query(
    q_id: int,
    i: int,
    q: pl.LazyFrame,
    run_config: RunConfig,
    args: argparse.Namespace,
    engine: None | pl.GPUEngine = None,
) -> pl.DataFrame:
    """Execute a query with NVTX annotation."""
    with nvtx.annotate(
        message=f"Query {q_id} - Iteration {i}",
        domain="cudf_polars",
        color="green",
    ):
        if run_config.executor == "cpu":
            return q.collect(engine="streaming")

        elif CUDF_POLARS_AVAILABLE:
            assert isinstance(engine, pl.GPUEngine)
            if args.debug:
                translator = Translator(q._ldf.visit(), engine)
                ir = translator.translate_ir()
                context = IRExecutionContext.from_config_options(
                    translator.config_options
                )
                if run_config.executor == "in-memory":
                    return ir.evaluate(
                        cache={}, timer=None, context=context
                    ).to_polars()
                elif run_config.executor == "streaming":
                    return evaluate_streaming(
                        ir,
                        translator.config_options,
                    )
                assert_never(run_config.executor)
            else:
                return q.collect(engine=engine)

        else:
            raise RuntimeError("The requested engine is not supported.")


def _query_type(num_queries: int) -> Callable[[str | int], list[int]]:
    def parse(query: str | int) -> list[int]:
        if isinstance(query, int):
            return [query]
        if query == "all":
            return list(range(1, num_queries + 1))

        result: set[int] = set()
        for part in query.split(","):
            if "-" in part:
                start, end = part.split("-")
                result.update(range(int(start), int(end) + 1))
            else:
                result.add(int(part))
        return sorted(result)

    return parse


def parse_args(
    args: Sequence[str] | None = None, num_queries: int = 22
) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="Cudf-Polars PDS-H Benchmarks",
        description="Experimental streaming-executor benchmarks.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "query",
        type=_query_type(num_queries),
        help=textwrap.dedent("""\
            Query to run. One of the following:
            - A single number (e.g. 11)
            - A comma-separated list of query numbers (e.g. 1,3,7)
            - A range of query number (e.g. 1-11,23-34)
            - The string 'all' to run all queries (1 through 22)"""),
    )
    parser.add_argument(
        "--path",
        type=str,
        default=os.environ.get("PDSH_DATASET_PATH"),
        help=textwrap.dedent("""\
            Path to the root directory of the PDS-H dataset.
            Defaults to the PDSH_DATASET_PATH environment variable."""),
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.environ.get("PDSH_DATASET_ROOT"),
        help="Root PDS-H dataset directory (ignored if --path is used).",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        help="Dataset scale factor.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".parquet",
        help=textwrap.dedent("""\
            File suffix for input table files.
            Default: .parquet"""),
    )
    parser.add_argument(
        "-e",
        "--executor",
        default="streaming",
        type=str,
        choices=["in-memory", "streaming", "cpu"],
        help=textwrap.dedent("""\
            Query executor backend:
                - in-memory : Evaluate query in GPU memory
                - streaming : Partitioned evaluation (default)
                - cpu       : Use Polars CPU engine"""),
    )
    parser.add_argument(
        "-c",
        "--cluster",
        default=None,
        type=str,
        choices=["single", "distributed"],
        help=textwrap.dedent("""\
            Cluster type to use with the 'streaming' executor.
                - single      : Run locally in a single process
                - distributed : Use Dask for multi-GPU execution"""),
    )
    parser.add_argument(
        "-s",
        "--scheduler",
        default=None,
        type=str,
        choices=["synchronous", "distributed"],
        help=textwrap.dedent("""\
            *Deprecated*: Use --cluster instead.

            Scheduler type to use with the 'streaming' executor.
                - synchronous : Run locally in a single process
                - distributed : Use Dask for multi-GPU execution"""),
    )
    parser.add_argument(
        "--runtime",
        type=str,
        choices=["tasks", "rapidsmpf"],
        default="tasks",
        help="Runtime to use for the streaming executor (tasks or rapidsmpf).",
    )
    parser.add_argument(
        "--stream-policy",
        type=str,
        choices=["auto", "default", "new", "pool"],
        default="auto",
        help=textwrap.dedent("""\
            CUDA stream policy (auto, default, new, pool).
            Default: auto (use the default policy for the runtime)"""),
    )
    parser.add_argument(
        "--n-workers",
        default=1,
        type=int,
        help="Number of Dask-CUDA workers (requires 'distributed' cluster).",
    )
    external_cluster_group = parser.add_mutually_exclusive_group()
    external_cluster_group.add_argument(
        "--scheduler-address",
        default=None,
        type=str,
        help=textwrap.dedent("""\
            Scheduler address for connecting to an existing Dask cluster.
            If provided, a cluster is not created and worker
            configuration options (--n-workers, --rmm-pool-size, etc.)
            are ignored since the workers are assumed to be started separately."""),
    )
    external_cluster_group.add_argument(
        "--scheduler-file",
        default=None,
        type=str,
        help=textwrap.dedent("""\
            Path to a scheduler file for connecting to an existing Dask cluster.
            If provided, a cluster is not created and worker
            configuration options (--n-workers, --rmm-pool-size, etc.)
            are ignored since the workers are assumed to be started separately."""),
    )
    parser.add_argument(
        "--blocksize",
        default=None,
        type=int,
        help="Target partition size, in bytes, for IO tasks.",
    )
    parser.add_argument(
        "--max-rows-per-partition",
        default=None,
        type=int,
        help="The maximum number of rows to process per partition.",
    )
    parser.add_argument(
        "--iterations",
        default=1,
        type=int,
        help="Number of times to run the same query.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Debug run.",
    )
    parser.add_argument(
        "--protocol",
        default="ucx",
        type=str,
        choices=["ucx"],
        help="Communication protocol to use for Dask: ucx (uses ucxx)",
    )
    parser.add_argument(
        "--shuffle",
        default=None,
        type=str,
        choices=[None, "rapidsmpf", "tasks"],
        help="Shuffle method to use for distributed execution.",
    )
    parser.add_argument(
        "--broadcast-join-limit",
        default=None,
        type=int,
        help="Set an explicit `broadcast_join_limit` option.",
    )
    parser.add_argument(
        "--threads",
        default=1,
        type=int,
        help="Number of threads to use on each GPU.",
    )
    parser.add_argument(
        "--rmm-pool-size",
        default=None,
        type=float,
        help=textwrap.dedent("""\
            Fraction of total GPU memory to allocate for RMM pool.
            Default: 0.5 (50%% of GPU memory) when --no-rmm-async,
                     None when --rmm-async"""),
    )
    parser.add_argument(
        "--rmm-release-threshold",
        default=None,
        type=float,
        help=textwrap.dedent("""\
            Passed to dask_cuda.LocalCUDACluster to control the release
            threshold for RMM pool memory.
            Default: None (no release threshold)"""),
    )
    parser.add_argument(
        "--rmm-async",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use RMM async memory resource. Note: only affects distributed cluster!",
    )
    parser.add_argument(
        "--rapidsmpf-oom-protection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use rapidsmpf CUDA managed memory-based OOM protection.",
    )
    parser.add_argument(
        "--rapidsmpf-dask-statistics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect rapidsmpf shuffle statistics. The output will be stored in the 'shuffle_stats' field of each record.",
    )
    parser.add_argument(
        "--rapidsmpf-print-statistics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print rapidsmpf shuffle statistics on each Dask worker upon completion.",
    )
    parser.add_argument(
        "--rapidsmpf-spill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use rapidsmpf for general spilling.",
    )
    parser.add_argument(
        "--spill-device",
        default=0.5,
        type=float,
        help="Rapidsmpf device spill threshold.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("at"),
        default="pdsh_results.jsonl",
        help="Output file path.",
    )
    parser.add_argument(
        "--summarize",
        action=argparse.BooleanOptionalAction,
        help="Summarize the results.",
        default=True,
    )
    parser.add_argument(
        "--print-results",
        action=argparse.BooleanOptionalAction,
        help="Print the query results",
        default=True,
    )
    parser.add_argument(
        "--explain",
        action=argparse.BooleanOptionalAction,
        help="Print an outline of the physical plan",
        default=False,
    )
    parser.add_argument(
        "--explain-logical",
        action=argparse.BooleanOptionalAction,
        help="Print an outline of the logical plan",
        default=False,
    )
    parser.add_argument(
        "--print-plans",
        action=argparse.BooleanOptionalAction,
        help="Print the query plans",
        default=True,
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Validate the result against CPU execution.",
    )
    parser.add_argument(
        "--baseline",
        choices=["duckdb", "cpu"],
        default="duckdb",
        help="Which engine to use as the baseline for validation.",
    )

    parser.add_argument(
        "--collect-traces",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect data tracing cudf-polars execution.",
    )

    parser.add_argument(
        "--stats-planning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable statistics planning.",
    )
    parser.add_argument(
        "--dynamic-planning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable dynamic shuffle planning (not yet implemented). ",
    )
    parser.add_argument(
        "--max-io-threads",
        default=2,
        type=int,
        help="Maximum number of IO threads for rapidsmpf runtime.",
    )
    parser.add_argument(
        "--native-parquet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use C++ read_parquet nodes for the rapidsmpf runtime.",
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=None,
        help="Optional directory to write query results as parquet files.",
    )
    parser.add_argument(
        "--spill-to-pinned-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=textwrap.dedent("""\
            Whether RapidsMPF should spill to pinned host memory when available,
            or use regular pageable host memory."""),
    )
    parser.add_argument(
        "--extra-info",
        type=json.loads,
        default={},
        help="Extra information to add to the output file (e.g. version information). Must be JSON-serializable.",
    )
    parser.add_argument(
        "--fallback-mode",
        type=str,
        choices=["warn", "raise", "silent"],
        default=None,
        help=textwrap.dedent("""\
            How to handle operations that don't support multiple partitions in streaming executor.
            - warn   : Emit a warning and fall back to single partition (default)
            - raise  : Raise an exception
            - silent : Silently fall back to single partition"""),
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.rmm_pool_size is None and not parsed_args.rmm_async:
        # The default rmm pool size depends on the rmm_async flag
        parsed_args.rmm_pool_size = 0.5

    return parsed_args


def run_polars(
    benchmark: Any,
    options: Sequence[str] | None = None,
    num_queries: int = 22,
) -> None:
    """Run the queries using the given benchmark and executor options."""
    args = parse_args(options, num_queries=num_queries)
    vars(args).update({"query_set": benchmark.name})
    run_config = RunConfig.from_args(args)
    validation_failures: list[int] = []
    query_failures: list[tuple[int, int]] = []

    client = initialize_dask_cluster(run_config, args)

    # Update n_workers from the actual cluster when using scheduler file/address
    if client is not None:
        actual_n_workers = client.scheduler_info()["n_workers"]
        run_config = dataclasses.replace(run_config, n_workers=actual_n_workers)

    records: defaultdict[int, list[Record]] = defaultdict(list)
    plans: dict[int, SerializablePlan] = {}
    engine: pl.GPUEngine | None = None

    if run_config.executor != "cpu":
        executor_options = get_executor_options(run_config, benchmark=benchmark)
        if run_config.runtime == "rapidsmpf":
            parquet_options = {
                "use_rapidsmpf_native": run_config.native_parquet,
            }
        else:
            parquet_options = {}
        engine = pl.GPUEngine(
            raise_on_fail=True,
            memory_resource=rmm.mr.CudaAsyncMemoryResource()
            if run_config.rmm_async
            else None,
            cuda_stream_policy=run_config.stream_policy,
            executor=run_config.executor,
            executor_options=executor_options,
            parquet_options=parquet_options,
        )

    for q_id in run_config.queries:
        try:
            q = getattr(benchmark, f"q{q_id}")(run_config)
        except AttributeError as err:
            raise NotImplementedError(f"Query {q_id} not implemented.") from err

        print_query_plan(
            q_id, q, args, run_config, engine, print_plans=args.print_plans
        )
        if (args.explain or args.explain_logical) and engine is not None:
            from cudf_polars.experimental.explain import serialize_query

            plans[q_id] = serialize_query(q, engine)

        records[q_id] = []
        for i in range(args.iterations):
            if _HAS_STRUCTLOG and run_config.collect_traces:
                setup_logging(q_id, i)
                if client is not None:
                    client.run(setup_logging, q_id, i)

            t0 = time.monotonic()

            try:
                result = execute_query(q_id, i, q, run_config, args, engine)
            except Exception:
                print(f"❌ query={q_id} iteration={i} failed!")
                print(traceback.format_exc())
                query_failures.append((q_id, i))
                continue
            if run_config.shuffle == "rapidsmpf" and run_config.gather_shuffle_stats:
                from rapidsmpf.integrations.dask.shuffler import (
                    clear_shuffle_statistics,
                    gather_shuffle_statistics,
                )

                shuffle_stats = gather_shuffle_statistics(client)
                clear_shuffle_statistics(client)
            else:
                shuffle_stats = None

            if args.validate and run_config.executor != "cpu":
                try:
                    assert_gpu_result_equal(
                        q,
                        engine=engine,
                        executor=run_config.executor,
                        check_exact=False,
                    )
                    print(f"✅ Query {q_id} passed validation!")
                except AssertionError as e:
                    validation_failures.append(q_id)
                    print(f"❌ Query {q_id} failed validation!\n{e}")

            t1 = time.monotonic()
            record = Record(
                query=q_id, iteration=i, duration=t1 - t0, shuffle_stats=shuffle_stats
            )
            if args.print_results:
                print(result)

            if args.results_directory is not None and i == 0:
                results_dir = Path(args.results_directory)
                results_dir.mkdir(parents=True, exist_ok=True)
                output_path = results_dir / f"q_{q_id:02d}.parquet"
                result.write_parquet(output_path)

            print(
                f"Query {q_id} - Iteration {i} finished in {record.duration:0.4f}s",
                flush=True,
            )
            records[q_id].append(record)

    run_config = dataclasses.replace(run_config, records=dict(records), plans=plans)

    # consolidate logs
    if _HAS_STRUCTLOG and run_config.collect_traces:

        def gather_logs() -> str:
            logger = logging.getLogger()
            return logger.handlers[0].stream.getvalue()  # type: ignore[attr-defined]

        if client is not None:
            # Gather logs from both client (for Query Plan) and workers
            worker_logs = "\n".join(client.run(gather_logs).values())
            client_logs = gather_logs()
            all_logs = client_logs + "\n" + worker_logs
        else:
            all_logs = gather_logs()

        parsed_logs = [json.loads(log) for log in all_logs.splitlines() if log]
        # Some other log records can end up in here. Filter those out.
        scope_values = {s.value for s in Scope}
        parsed_logs = [log for log in parsed_logs if log.get("scope") in scope_values]
        # Now we want to augment the existing Records with the trace data.

        def group_key(x: dict) -> int:
            return x["query_id"]

        def sort_key(x: dict) -> tuple[int, int]:
            return x["query_id"], x["iteration"]

        grouped = itertools.groupby(
            sorted(parsed_logs, key=sort_key),
            key=group_key,
        )

        for query_id, run_logs_group in grouped:
            run_logs = list(run_logs_group)
            by_iteration = [
                list(x)
                for _, x in itertools.groupby(run_logs, key=lambda x: x["iteration"])
            ]
            run_records = run_config.records[query_id]
            assert len(by_iteration) == len(run_records)  # same number of iterations
            all_traces = [list(iteration) for iteration in by_iteration]

            new_records = [
                dataclasses.replace(record, traces=traces)
                for record, traces in zip(run_records, all_traces, strict=True)
            ]

            run_config.records[query_id] = new_records

    if args.summarize:
        run_config.summarize()

    if client is not None:
        client.close(timeout=60)

    if args.validate and run_config.executor != "cpu":
        print("\nValidation Summary")
        print("==================")
        if validation_failures:
            print(
                f"{len(validation_failures)} queries failed validation: {sorted(set(validation_failures))}"
            )
        else:
            print("All validated queries passed.")

    args.output.write(json.dumps(run_config.serialize(engine=engine)))
    args.output.write("\n")

    if query_failures or validation_failures:
        sys.exit(1)


def setup_logging(query_id: int, iteration: int) -> None:  # noqa: D103
    import cudf_polars.dsl.tracing

    if not cudf_polars.dsl.tracing.LOG_TRACES:
        msg = (
            "Tracing requested via --collect-traces, but tracking is not enabled. "
            "Verify that 'CUDF_POLARS_LOG_TRACES' is set and structlog is installed."
        )
        raise RuntimeError(msg)

    if _HAS_STRUCTLOG:
        # structlog uses contextvars to propagate context down to where log records
        # are emitted. Ideally, we'd just set the contextvars here using
        # structlog.bind_contextvars; for the distributed cluster we would need
        # to use something like client.run to set the contextvars on the worker.
        # However, there's an unfortunate conflict between structlog's use of
        # context vars and how Dask Workers actually execute tasks, such that
        # the contextvars set via `client.run` aren't visible to the actual
        # tasks.
        #
        # So instead we make a new logger each time we need a new context,
        # i.e. for each query/iteration pair.

        def make_injector(
            query_id: int, iteration: int
        ) -> Callable[[logging.Logger, str, dict[str, Any]], dict[str, Any]]:
            def inject(
                logger: Any, method_name: Any, event_dict: Any
            ) -> dict[str, Any]:
                event_dict["query_id"] = query_id
                event_dict["iteration"] = iteration
                return event_dict

            return inject

        shared_processors = [
            structlog.contextvars.merge_contextvars,
            make_injector(query_id, iteration),
            structlog.processors.add_log_level,
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.PROCESS,
                    structlog.processors.CallsiteParameter.THREAD,
                ],
            ),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S.%f", utc=False),
        ]

        # For logging to a file
        json_renderer = structlog.processors.JSONRenderer()

        stream = io.StringIO()
        json_file_handler = logging.StreamHandler(stream)
        json_file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=json_renderer,
                foreign_pre_chain=shared_processors,
            )
        )

        logging.basicConfig(level=logging.INFO, handlers=[json_file_handler])

        structlog.configure(
            processors=[
                *shared_processors,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            cache_logger_on_first_use=True,
        )


PDSDS_TABLE_NAMES: list[str] = [
    "call_center",
    "catalog_page",
    "catalog_returns",
    "catalog_sales",
    "customer",
    "customer_address",
    "customer_demographics",
    "date_dim",
    "household_demographics",
    "income_band",
    "inventory",
    "item",
    "promotion",
    "reason",
    "ship_mode",
    "store",
    "store_returns",
    "store_sales",
    "time_dim",
    "warehouse",
    "web_page",
    "web_returns",
    "web_sales",
    "web_site",
]

PDSH_TABLE_NAMES: list[str] = [
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
]


def print_duckdb_plan(
    q_id: int,
    sql: str,
    dataset_path: Path,
    suffix: str,
    query_set: str,
    args: argparse.Namespace,
) -> None:
    """Print DuckDB query plan using EXPLAIN."""
    if duckdb is None:
        raise ImportError(duckdb_err)

    if query_set == "pdsds":
        tbl_names = PDSDS_TABLE_NAMES
    else:
        tbl_names = PDSH_TABLE_NAMES

    with duckdb.connect() as conn:
        for name in tbl_names:
            pattern = (Path(dataset_path) / name).as_posix() + suffix
            conn.execute(
                f"CREATE OR REPLACE VIEW {name} AS "
                f"SELECT * FROM parquet_scan('{pattern}');"
            )

        if args.explain_logical and args.explain:
            conn.execute("PRAGMA explain_output = 'all';")
        elif args.explain_logical:
            conn.execute("PRAGMA explain_output = 'optimized_only';")
        else:
            conn.execute("PRAGMA explain_output = 'physical_only';")

        print(f"\nDuckDB Query {q_id} - Plan\n")

        plan_rows = conn.execute(f"EXPLAIN {sql}").fetchall()
        for _, line in plan_rows:
            print(line)


def execute_duckdb_query(
    query: str,
    dataset_path: Path,
    *,
    suffix: str = ".parquet",
    query_set: str = "pdsh",
) -> pl.DataFrame:
    """Execute a query with DuckDB."""
    if duckdb is None:
        raise ImportError(duckdb_err)
    if query_set == "pdsds":
        tbl_names = PDSDS_TABLE_NAMES
    else:
        tbl_names = PDSH_TABLE_NAMES
    with duckdb.connect() as conn:
        for name in tbl_names:
            pattern = (Path(dataset_path) / name).as_posix() + suffix
            conn.execute(
                f"CREATE OR REPLACE VIEW {name} AS "
                f"SELECT * FROM parquet_scan('{pattern}');"
            )
        return conn.execute(query).pl()


def run_duckdb(
    duckdb_queries_cls: Any, options: Sequence[str] | None = None, *, num_queries: int
) -> None:
    """Run the benchmark with DuckDB."""
    args = parse_args(options, num_queries=num_queries)
    vars(args).update({"query_set": duckdb_queries_cls.name})
    run_config = RunConfig.from_args(args)
    records: defaultdict[int, list[Record]] = defaultdict(list)

    for q_id in run_config.queries:
        try:
            get_q = getattr(duckdb_queries_cls, f"q{q_id}")
        except AttributeError as err:
            raise NotImplementedError(f"Query {q_id} not implemented.") from err

        sql = get_q(run_config)

        if args.explain or args.explain_logical:
            print_duckdb_plan(
                q_id=q_id,
                sql=sql,
                dataset_path=run_config.dataset_path,
                suffix=run_config.suffix,
                query_set=duckdb_queries_cls.name,
                args=args,
            )

        print(f"DuckDB Executing: {q_id}")
        records[q_id] = []

        for i in range(args.iterations):
            t0 = time.time()
            result = execute_duckdb_query(
                sql,
                run_config.dataset_path,
                suffix=run_config.suffix,
                query_set=duckdb_queries_cls.name,
            )
            t1 = time.time()
            record = Record(query=q_id, iteration=i, duration=t1 - t0)
            if args.print_results:
                print(result)
            print(f"Query {q_id} - Iteration {i} finished in {record.duration:0.4f}s")
            records[q_id].append(record)

    run_config = dataclasses.replace(run_config, records=dict(records))
    if args.summarize:
        run_config.summarize()


def run_validate(
    polars_queries_cls: Any,
    duckdb_queries_cls: Any,
    options: Sequence[str] | None = None,
    *,
    num_queries: int,
    check_dtypes: bool,
    check_column_order: bool,
) -> None:
    """Validate Polars CPU/GPU vs DuckDB."""
    from polars.testing import assert_frame_equal

    args = parse_args(options, num_queries=num_queries)
    vars(args).update({"query_set": polars_queries_cls.name})
    run_config = RunConfig.from_args(args)

    baseline = args.baseline
    if baseline not in {"duckdb", "cpu"}:
        raise ValueError("Baseline must be one of: 'duckdb', 'cpu'")

    failures: list[int] = []

    engine: pl.GPUEngine | None = None
    if run_config.executor != "cpu":
        engine = pl.GPUEngine(
            raise_on_fail=True,
            executor=run_config.executor,
            executor_options=get_executor_options(run_config, polars_queries_cls),
        )

    for q_id in run_config.queries:
        print(f"\nValidating Query {q_id}")
        try:
            get_pl = getattr(polars_queries_cls, f"q{q_id}")
            get_ddb = getattr(duckdb_queries_cls, f"q{q_id}")
        except AttributeError as err:
            raise NotImplementedError(f"Query {q_id} not implemented.") from err

        polars_query = get_pl(run_config)
        if baseline == "duckdb":
            base_sql = get_ddb(run_config)
            base_result = execute_duckdb_query(
                base_sql,
                run_config.dataset_path,
                query_set=duckdb_queries_cls.name,
            )
        else:
            base_result = polars_query.collect(engine="streaming")

        if run_config.executor == "cpu":
            test_result = polars_query.collect(engine="streaming")
        else:
            try:
                test_result = polars_query.collect(engine=engine)
            except Exception as e:
                failures.append(q_id)
                print(f"❌ Query {q_id} failed validation: GPU execution failed.\n{e}")
                continue

        try:
            assert_frame_equal(
                base_result,
                test_result,
                check_dtypes=check_dtypes,
                check_column_order=check_column_order,
            )
            print(f"✅ Query {q_id} passed validation.")
        except AssertionError as e:
            failures.append(q_id)
            print(f"❌ Query {q_id} failed validation:\n{e}")
            if args.print_results:
                print("Baseline Result:\n", base_result)
                print("Test Result:\n", test_result)

    if failures:
        print("\nValidation Summary:")
        print("===================")
        print(f"{len(failures)} query(s) failed: {failures}")
    else:
        print("\nAll queries passed validation.")
