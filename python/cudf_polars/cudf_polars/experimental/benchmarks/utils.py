# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions/classes for running the PDS-H and PDS-DS benchmarks."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import statistics
import sys
import textwrap
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, assert_never

import nvtx

import polars as pl

try:
    import pynvml
except ImportError:
    pynvml = None

try:
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


ExecutorType = Literal["in-memory", "streaming", "cpu"]


@dataclasses.dataclass
class Record:
    """Results for a single run of a single PDS-H query."""

    query: int
    duration: float


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
            except (AttributeError, ImportError):  # noqa: PERF203
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


def _infer_scale_factor(path: str | Path, suffix: str) -> int | float:
    name = Path(sys.argv[0]).name

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
    """Results for a PDS-H query run."""

    queries: list[int]
    suffix: str
    executor: ExecutorType
    scheduler: str
    n_workers: int
    versions: PackageVersions = dataclasses.field(
        default_factory=PackageVersions.collect
    )
    records: dict[int, list[Record]] = dataclasses.field(default_factory=dict)
    dataset_path: Path
    scale_factor: int | float
    shuffle: str | None = None
    broadcast_join_limit: int | None = None
    blocksize: int | None = None
    max_rows_per_partition: int | None = None
    threads: int
    iterations: int
    timestamp: str = dataclasses.field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    hardware: HardwareInfo = dataclasses.field(default_factory=HardwareInfo.collect)
    rmm_async: bool
    rapidsmpf_oom_protection: bool
    rapidsmpf_spill: bool
    spill_device: float

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RunConfig:
        """Create a RunConfig from command line arguments."""
        executor: ExecutorType = args.executor
        scheduler = args.scheduler

        if executor == "in-memory" or executor == "cpu":
            scheduler = None

        path = args.path
        if (scale_factor := args.scale) is None:
            if path is None:
                raise ValueError(
                    "Must specify --root and --scale if --path is not specified."
                )
            scale_factor = _infer_scale_factor(path, args.suffix)
        if path is None:
            path = f"{args.root}/scale-{scale_factor}"
        try:
            scale_factor = int(scale_factor)
        except ValueError:
            scale_factor = float(scale_factor)

        if args.scale is not None:
            # Validate the user-supplied scale factor
            sf_inf = _infer_scale_factor(path, args.suffix)
            rel_error = abs((scale_factor - sf_inf) / sf_inf)
            if rel_error > 0.01:
                raise ValueError(
                    f"Specified scale factor is {args.scale}, "
                    f"but the inferred scale factor is {sf_inf}."
                )

        return cls(
            queries=args.query,
            executor=executor,
            scheduler=scheduler,
            n_workers=args.n_workers,
            shuffle=args.shuffle,
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
        )

    def serialize(self, engine: pl.GPUEngine | None) -> dict:
        """Serialize the run config to a dictionary."""
        result = dataclasses.asdict(self)

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
            if self.executor == "streaming":
                print(f"scheduler: {self.scheduler}")
                print(f"blocksize: {self.blocksize}")
                print(f"shuffle_method: {self.shuffle}")
                print(f"broadcast_join_limit: {self.broadcast_join_limit}")
                if self.scheduler == "distributed":
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

    if run_config.blocksize:
        executor_options["target_partition_size"] = run_config.blocksize
    if run_config.max_rows_per_partition:
        executor_options["max_rows_per_partition"] = run_config.max_rows_per_partition
    if run_config.shuffle:
        executor_options["shuffle_method"] = run_config.shuffle
    if run_config.broadcast_join_limit:
        executor_options["broadcast_join_limit"] = run_config.broadcast_join_limit
    if run_config.rapidsmpf_spill:
        executor_options["rapidsmpf_spill"] = run_config.rapidsmpf_spill
    if run_config.scheduler == "distributed":
        executor_options["scheduler"] = "distributed"

    if (
        benchmark
        and benchmark.__name__ == "PDSHQueries"
        and run_config.executor == "streaming"
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
) -> None:
    """Print the query plan."""
    if run_config.executor == "cpu":
        if args.explain_logical:
            print(f"\nQuery {q_id} - Logical plan\n")
            print(q.explain())
        elif args.explain:
            print(f"\nQuery {q_id} - Physical plan\n")
            print(q.show_graph(engine="streaming", plan_stage="physical"))
    elif CUDF_POLARS_AVAILABLE:
        assert isinstance(engine, pl.GPUEngine)
        if args.explain_logical:
            print(f"\nQuery {q_id} - Logical plan\n")
            print(explain_query(q, engine, physical=False))
        elif args.explain:
            print(f"\nQuery {q_id} - Physical plan\n")
            print(explain_query(q, engine))
    else:
        raise RuntimeError(
            "Cannot provide the logical or physical plan because cudf_polars is not installed."
        )


def initialize_dask_cluster(run_config: RunConfig, args: argparse.Namespace):  # type: ignore
    """Initialize a Dask distributed cluster."""
    if run_config.scheduler != "distributed":
        return None

    from dask_cuda import LocalCUDACluster
    from distributed import Client

    kwargs = {
        "n_workers": run_config.n_workers,
        "dashboard_address": ":8585",
        "protocol": args.protocol,
        "rmm_pool_size": args.rmm_pool_size,
        "rmm_async": args.rmm_async,
        "threads_per_worker": run_config.threads,
    }

    # Avoid UVM in distributed cluster
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
                        "dask_statistics": str(args.rapidsmpf_oom_protection),
                    }
                ),
            )
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
                if run_config.executor == "in-memory":
                    return ir.evaluate(cache={}, timer=None).to_polars()
                elif run_config.executor == "streaming":
                    return evaluate_streaming(ir, translator.config_options).to_polars()
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
        "-s",
        "--scheduler",
        default="synchronous",
        type=str,
        choices=["synchronous", "distributed"],
        help=textwrap.dedent("""\
            Scheduler type to use with the 'streaming' executor.
                - synchronous : Run locally in a single process
                - distributed : Use Dask for multi-GPU execution"""),
    )
    parser.add_argument(
        "--n-workers",
        default=1,
        type=int,
        help="Number of Dask-CUDA workers (requires 'distributed' scheduler).",
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
        choices=["ucx-old", "ucx"],
        help="Communication protocol to use for Dask: ucx (uses ucxx) or ucx-old (uses ucx-py)",
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
        default=0.5,
        type=float,
        help=textwrap.dedent("""\
            Fraction of total GPU memory to allocate for RMM pool.
            Default: 0.5 (50%% of GPU memory)"""),
    )
    parser.add_argument(
        "--rmm-async",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use RMM async memory resource.",
    )
    parser.add_argument(
        "--rapidsmpf-oom-protection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use rapidsmpf CUDA managed memory-based OOM protection.",
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
    return parser.parse_args(args)


def run_polars(
    benchmark: Any,
    options: Sequence[str] | None = None,
    num_queries: int = 22,
) -> None:
    """Run the queries using the given benchmark and executor options."""
    args = parse_args(options, num_queries=num_queries)
    run_config = RunConfig.from_args(args)
    validation_failures: list[int] = []

    client = initialize_dask_cluster(run_config, args)  # type: ignore

    records: defaultdict[int, list[Record]] = defaultdict(list)
    engine: pl.GPUEngine | None = None

    if run_config.executor != "cpu":
        executor_options = get_executor_options(run_config, benchmark=benchmark)
        engine = pl.GPUEngine(
            raise_on_fail=True,
            executor=run_config.executor,
            executor_options=executor_options,
        )

    for q_id in run_config.queries:
        try:
            q = getattr(benchmark, f"q{q_id}")(run_config)
        except AttributeError as err:
            raise NotImplementedError(f"Query {q_id} not implemented.") from err

        print_query_plan(q_id, q, args, run_config, engine)

        records[q_id] = []

        for i in range(args.iterations):
            t0 = time.monotonic()

            result = execute_query(q_id, i, q, run_config, args, engine)

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
            record = Record(query=q_id, duration=t1 - t0)
            if args.print_results:
                print(result)

            print(f"Query {q_id} - Iteration {i} finished in {record.duration:0.4f}s")
            records[q_id].append(record)

    run_config = dataclasses.replace(run_config, records=dict(records))

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
