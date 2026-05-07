# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark utilities for the RapidsMPF SPMD and Ray frontends."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import io
import itertools
import json
import logging
import os
import pprint
import sys
import textwrap
import time
import traceback
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any, Literal, assert_never

import nvtx

import polars as pl

__all__: list[str] = [
    "COUNT_DTYPE",
    "QueryResult",
    "RunConfig",
    "build_parser",
    "get_data",
    "parse_args",
    "run_duckdb",
    "run_polars",
]

# The dtype for count() aggregations depends on the presence
# of the polars-runtime-64 package (`polars[rt64]`).
HAS_POLARS_RT_64 = pl.config.plr.RUNTIME_REPR == "rt64"
COUNT_DTYPE = pl.UInt64() if HAS_POLARS_RT_64 else pl.UInt32()

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
    from cudf_polars.experimental.benchmarks.asserts import (
        ValidationError,
        assert_tpch_result_equal,
    )
    from cudf_polars.experimental.explain import explain_query
    from cudf_polars.experimental.parallel import evaluate_streaming
    from cudf_polars.utils.config import ConfigOptions

    CUDF_POLARS_AVAILABLE = True
except ImportError:
    CUDF_POLARS_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Callable

    from cudf_polars.experimental.explain import SerializablePlan
    from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine
    from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
POLARS_VALIDATION_OPTIONS = {
    "check_row_order": True,
    "check_column_order": True,
    "check_dtypes": True,
    "check_exact": False,
    "rel_tol": 1e-5,
    "abs_tol": 1e-2,
}


def get_validation_options(args: Any) -> dict[str, Any]:
    """Get validation options dict from parsed arguments."""
    return {
        **POLARS_VALIDATION_OPTIONS,
        "abs_tol": args.validation_abs_tol,
    }


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
class ValidationResult:
    """
    Result of a validation run.

    Parameters
    ----------
    status
        The status of the validation. Either 'Passed' or 'Failed'.
    message
        The message from the validation. This should be ``None`` if
        the validation passed, and a string describing the failure otherwise.
    details
        Additional details about the validation failure.
    """

    status: Literal["Passed", "Failed"]
    message: str | None
    details: dict[str, Any] | None = None

    @classmethod
    def from_error(cls, error: Exception) -> ValidationResult:
        """
        Create a ValidationResult from some exception.

        Parameters
        ----------
        error : Exception
            The error to create a ValidationResult from.

            This will correctly propagate "message" and "details" from
            ``cudf_polars.testing.asserts.ValidationError``.

        Returns
        -------
        ValidationResult
            The ValidationResult created from the error.
        """
        match error:
            case ValidationError(message=message, details=details):
                return cls(status="Failed", message=message, details=details)
            case _:
                return cls(status="Failed", message=str(error))


@dataclasses.dataclass
class ValidationMethod:
    """
    Information about how the validation was performed.

    Parameters
    ----------
    expected_source
        A name indicating the source of the expected results.

        - 'polars-cpu': Run polars against the same data
        - 'duckdb': Compare against pre-computed DuckDB results

    comparison_method
        How the comparison was performed. Currently, only
        'polars' is supported, which indicates that ``polars.testing.assert_frame_equal``
        was used.

    comparison_options
        Additional options passed to the comparison method, controlling
        things like the tolerance for floating point comparisons.
    """

    expected_source: Literal["polars-cpu", "duckdb"]
    comparison_method: Literal["polars"]
    comparison_options: dict[str, Any]


@dataclasses.dataclass(kw_only=True)
class FailedRecord:
    """Records a failed query iteration."""

    query: int
    iteration: int
    status: Literal["error"] = "error"
    traceback: str


@dataclasses.dataclass(kw_only=True)
class SuccessRecord:
    """Results for a single run of a single PDS-H query."""

    query: int
    iteration: int
    duration: float
    statistics: dict[str, Any] | None = None
    traces: list[dict[str, Any]] | None = None
    validation_result: ValidationResult | None = None
    status: Literal["success"] = "success"

    @classmethod
    def new(
        cls,
        query: int,
        iteration: int,
        duration: float,
        statistics: dict[str, Any] | None = None,
        traces: list[dict[str, Any]] | None = None,
    ) -> SuccessRecord:
        """Create a Record from plain data."""
        return cls(
            query=query,
            iteration=iteration,
            duration=duration,
            statistics=statistics,
            traces=traces,
        )


@dataclasses.dataclass
class QueryRunResult:
    """Result of running a single query (all iterations)."""

    query_records: list[SuccessRecord | FailedRecord]
    plan: SerializablePlan | None
    iteration_failures: list[tuple[int, int]]
    validation_failed: bool


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
    duckdb: str | None

    @classmethod
    def collect(cls) -> PackageVersions:
        """Collect the versions of the software used to run the query."""
        packages = [
            "cudf_polars",
            "duckdb",
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


def get_data(path: str | Path, table_name: str, suffix: str = "") -> pl.LazyFrame:
    """
    Get table from dataset.

    Supports both single-file (e.g. ``supplier.parquet``) and
    directory-based (e.g. ``supplier/``) parquet layouts.
    When the file ``{path}/{table_name}{suffix}`` does not exist on the
    local filesystem, falls back to scanning ``{path}/{table_name}`` as a
    directory of parquet files.
    """
    file_path = Path(path) / f"{table_name}{suffix}"
    if suffix and not file_path.exists():
        # Directory-based layout: e.g. tpch-rs partitioned output
        return pl.scan_parquet(Path(path) / table_name)
    return pl.scan_parquet(file_path)


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
    """Benchmark run configuration for SPMD / Ray / DuckDB frontends."""

    engine_name: Literal["polars-cpu", "cudf-polars", "duckdb"]
    # Query selection & dataset
    queries: list[int]
    query_set: str
    dataset_path: Path
    scale_factor: int | float
    suffix: str
    qualification: bool = False

    # Execution mode
    executor: ExecutorType  # "in-memory" | "streaming" | "cpu"
    frontend: str  # "spmd" | "ray" | "duckdb"
    connect: str | None = None
    num_gpus: int | None = None

    # Run parameters
    iterations: int
    io_mode: Literal["cold", "lukewarm", "hot"] = "lukewarm"
    collect_traces: bool = False
    native_parquet: bool = True
    max_io_threads: int = 2
    # All streaming/rapidsmpf/engine knobs
    streaming_options: StreamingOptions = dataclasses.field(
        default_factory=lambda: __import__(
            "cudf_polars.experimental.rapidsmpf.frontend.options",
            fromlist=["StreamingOptions"],
        ).StreamingOptions()
    )

    # Validation
    validation_method: ValidationMethod | None = None

    # DuckDB configuration
    duckdb_threads: int | None = None
    duckdb_memory_limit: str | None = None
    duckdb_temp_dir: str | None = None

    # Metadata / output (populated at runtime)
    n_workers: int = 1
    extra_info: dict[str, Any] = dataclasses.field(default_factory=dict)
    versions: PackageVersions = dataclasses.field(
        default_factory=PackageVersions.collect
    )
    records: dict[int, list[SuccessRecord | FailedRecord]] = dataclasses.field(
        default_factory=dict
    )
    plans: dict[int, Any] = dataclasses.field(default_factory=dict)
    hardware: HardwareInfo = dataclasses.field(default_factory=HardwareInfo.collect)
    run_id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)
    timestamp: str = dataclasses.field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    def __post_init__(self) -> None:  # noqa: D105
        if self.io_mode == "hot" and self.iterations < 2:
            raise ValueError(
                "--io-mode hot requires at least 2 iterations: "
                "iteration 0 warms the cache, iterations 1+ are the hot measurements."
            )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RunConfig:
        """Create a RunConfig from command line arguments."""
        from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions

        streaming_options = StreamingOptions._from_argparse(args)

        path = args.path
        name = args.query_set
        scale_factor = args.scale

        if args.qualification and "pdsds" not in name:
            raise ValueError("--qualification can only be used with PDS-DS benchmarks.")

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

        if args.validate_directory:
            validation_method = ValidationMethod(
                expected_source="duckdb",
                comparison_method="polars",
                comparison_options=get_validation_options(args),
            )
        elif args.validate:
            validation_method = ValidationMethod(
                expected_source="polars-cpu" if args.baseline == "cpu" else "duckdb",
                comparison_method="polars",
                comparison_options=get_validation_options(args),
            )
        else:
            validation_method = None

        engine_name: Literal["polars-cpu", "cudf-polars", "duckdb"]
        if args.engine == "duckdb":
            engine_name = "duckdb"
        elif args.engine == "polars":
            if args.executor == "cpu":
                engine_name = "polars-cpu"
            else:
                engine_name = "cudf-polars"
        else:
            raise ValueError(f"Invalid engine: {args.engine}")

        return cls(
            engine_name=engine_name,
            queries=args.query,
            query_set=name,
            dataset_path=path,
            scale_factor=scale_factor,
            suffix=args.suffix,
            qualification=args.qualification,
            executor=args.executor,
            frontend=args.frontend,
            iterations=args.iterations,
            io_mode=args.io_mode,
            collect_traces=args.collect_traces,
            native_parquet=args.native_parquet,
            max_io_threads=args.max_io_threads,
            streaming_options=streaming_options,
            connect=args.connect,
            num_gpus=args.num_gpus,
            validation_method=validation_method,
            extra_info=args.extra_info,
            duckdb_threads=args.duckdb_threads,
            duckdb_memory_limit=args.duckdb_memory_limit,
            duckdb_temp_dir=args.duckdb_temp_dir,
        )

    def serialize(self, engine: pl.GPUEngine | None) -> dict:
        """Serialize the run config to a dictionary."""
        opts = self.streaming_options
        result: dict[str, Any] = {
            "engine_name": self.engine_name,
            "queries": self.queries,
            "query_set": self.query_set,
            "dataset_path": str(self.dataset_path),
            "scale_factor": self.scale_factor,
            "suffix": self.suffix,
            "qualification": self.qualification,
            "executor": self.executor,
            "frontend": self.frontend,
            "iterations": self.iterations,
            "io_mode": self.io_mode,
            "collect_traces": self.collect_traces,
            "native_parquet": self.native_parquet,
            "max_io_threads": self.max_io_threads,
            "n_workers": self.n_workers,
            "extra_info": self.extra_info,
            "run_id": str(self.run_id),
            "timestamp": self.timestamp,
            "streaming_options": {
                "rapidsmpf": opts.to_rapidsmpf_options().get_strings(),
                "executor": opts.to_executor_options(),
                "engine": {k: str(v) for k, v in opts.to_engine_options().items()},
            },
            "records": {
                k: [dataclasses.asdict(r) for r in v] for k, v in self.records.items()
            },
            "plans": {},
            "versions": dataclasses.asdict(self.versions),
            "hardware": dataclasses.asdict(self.hardware),
            "validation_method": dataclasses.asdict(self.validation_method)
            if self.validation_method
            else None,
        }
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
            print(f"frontend: {self.frontend}")
            if self.executor == "streaming":
                opts = self.streaming_options.to_executor_options()
                print(f"native_parquet: {self.native_parquet}")
                print(f"n_workers: {self.n_workers}")
                print(f"target_partition_size: {opts.get('target_partition_size')}")
                print(f"broadcast_join_limit: {opts.get('broadcast_join_limit')}")
                print(f"dynamic_planning: {opts.get('dynamic_planning', 'default')}")
            valid_durations = [
                record.duration for record in records if record.status == "success"
            ]
            if len(valid_durations) > 0:
                print(f"iterations: {self.iterations}")
                print("---------------------------------------")
                print(f"min time : {min(valid_durations):0.4f}")
                print(f"max time : {max(valid_durations):0.4f}")
                print(f"mean time: {mean(valid_durations):0.4f}")
                print("=======================================")
        any_success = any(record.status == "success" for record in records)

        if any_success:
            total_mean_time = sum(
                mean(
                    record.duration for record in records if record.status == "success"
                )
                for records in self.records.values()
                if records
            )
            print(f"Total mean time across all queries: {total_mean_time:.4f} seconds")
        else:
            print("No successful queries")


def get_executor_options(
    run_config: RunConfig, benchmark: Any = None
) -> dict[str, Any]:
    """Generate executor_options for GPUEngine."""
    executor_options: dict[str, Any] = (
        run_config.streaming_options.to_executor_options()
    )
    executor_options["max_io_threads"] = run_config.max_io_threads

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


def drop_file_page_cache_recursively(path: os.PathLike | str) -> None:
    """Drop the Linux page cache for all files under `path`."""
    try:
        import kvikio
    except ImportError as err:
        raise RuntimeError(
            "kvikio is required for cold-run page cache dropping. "
            "Install it or switch to --io-mode lukewarm."
        ) from err
    p = Path(path).expanduser()
    if p.is_file():
        kvikio.drop_file_page_cache(p)
        return
    for f in p.rglob("*"):
        if f.is_file():
            kvikio.drop_file_page_cache(f)


def execute_query(
    q_id: int,
    i: int,
    q: pl.LazyFrame,
    run_config: RunConfig,
    args: argparse.Namespace,
    engine: None | pl.GPUEngine = None,
) -> tuple[pl.DataFrame, float]:
    """Execute a query with NVTX annotation."""
    if run_config.io_mode == "cold":
        drop_file_page_cache_recursively(run_config.dataset_path)

    with nvtx.annotate(
        message=f"Query {q_id} - Iteration {i}",
        domain="cudf_polars",
        color="green",
    ):
        if run_config.executor == "cpu":
            t0 = time.monotonic()
            result = q.collect(engine="streaming")
            t1 = time.monotonic()

        elif CUDF_POLARS_AVAILABLE:
            assert isinstance(engine, pl.GPUEngine)
            if args.debug:
                translator = Translator(q._ldf.visit(), engine)
                ir = translator.translate_ir()
                context = IRExecutionContext()
                if run_config.executor == "in-memory":
                    t0 = time.monotonic()
                    result = ir.evaluate(
                        cache={}, timer=None, context=context
                    ).to_polars()
                    t1 = time.monotonic()
                elif run_config.executor == "streaming":
                    t0 = time.monotonic()
                    result = evaluate_streaming(
                        ir,
                        translator.config_options,
                    )
                    t1 = time.monotonic()
                else:
                    assert_never(run_config.executor)
            else:
                t0 = time.monotonic()
                result = q.collect(engine=engine)
                t1 = time.monotonic()

        else:
            raise RuntimeError("The requested engine is not supported.")

        return result, t1 - t0


def validate_result(
    result: pl.DataFrame,
    expected: pl.DataFrame,
    sort_by: list[tuple[str, bool]],
    limit: int | None = None,
    sort_keys: list[tuple[pl.Expr, bool]] | None = None,
    **kwargs: Any,
) -> ValidationResult:
    """
    Validate the computed result against the expected answer.

    This takes care of special handling for validating TPC-H queries,
    where multiple results might be considered correct.

    See Also
    --------
    cudf_polars.testing.asserts.assert_tpch_result_equal
    """
    try:
        assert_tpch_result_equal(
            result,
            expected,
            sort_by=sort_by,
            limit=limit,
            sort_keys=sort_keys,
            **kwargs,
        )
    except Exception as e:
        return ValidationResult.from_error(e)
    else:
        return ValidationResult(status="Passed", message=None)


@dataclasses.dataclass
class QueryResult:
    """
    Representation of a query's result.

    Parameters
    ----------
    frame: pl.LazyFrame
        The result of the query.
    sort_by: list[tuple[str, bool]]
        The columns that the query sorts by. Each tuple contains (column_name, descending_flag).
        Used for the ties/limit boundary logic in validation.
    sort_keys: list[tuple[pl.Expr, bool]] | None
        Optional Polars expressions for the sortedness check. Use this when the query
        sorts by a conditional expression (e.g. ``CASE WHEN lochierarchy = 0 THEN i_category END``)
        that cannot be represented as a plain column name in ``sort_by``. When provided,
        these expressions are evaluated against the output and used only for the sortedness
        check; ``sort_by`` still drives the ties/limit boundary logic.
    limit: int | None
        The limit of the query, if any.

    """

    frame: pl.LazyFrame
    sort_by: list[tuple[str, bool]]
    limit: int | None = None
    nulls_last: bool = True
    sort_keys: list[tuple[pl.Expr, bool]] | None = None


def _collect_statistics(engine: StreamingEngine | None) -> dict[str, Any] | None:
    """Gather + clear per-rank rapidsmpf statistics into a merged dict."""
    return None if engine is None else engine.global_statistics(clear=True).to_dict()


def run_polars_query_iteration(
    q_id: int,
    iteration: int,
    q: pl.LazyFrame,
    run_config: RunConfig,
    args: argparse.Namespace,
    engine: StreamingEngine | None,
    expected: pl.DataFrame | None,
    query_result: Any,
    prepare_validation_result: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    result_casts: list[pl.Expr] | None = None,
) -> SuccessRecord:
    """Run a single query iteration. Caller must wrap in try/except."""
    result, duration = execute_query(q_id, iteration, q, run_config, args, engine)

    if expected is not None and prepare_validation_result is not None:
        result = prepare_validation_result(result)

    if expected is not None and result_casts:
        # Applying the casts to the polars result is
        # a workaround we need because of a polars bug
        # See https://github.com/pola-rs/polars/issues/27269
        # Once we support polars 1.40, we should remove this
        result = result.with_columns(*result_casts)

    statistics = _collect_statistics(engine)

    if expected is not None:
        validation_result = validate_result(
            result,
            expected,
            query_result.sort_by,
            limit=query_result.limit,
            nulls_last=query_result.nulls_last,
            sort_keys=query_result.sort_keys,
            **get_validation_options(args),
        )
    else:
        validation_result = None

    if args.print_results:
        print(result)

    if args.results_directory is not None and iteration == 0:
        results_dir = Path(args.results_directory)
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / f"q_{q_id:02d}.parquet"
        result.write_parquet(output_path)

    return SuccessRecord(
        query=q_id,
        iteration=iteration,
        duration=duration,
        statistics=statistics,
        validation_result=validation_result,
    )


def run_polars_query(
    q_id: int,
    benchmark: Any,
    run_config: RunConfig,
    args: argparse.Namespace,
    engine: StreamingEngine | None,
    numeric_type: str,
    date_type: str,
    validation_files: dict[int, Path] | None,
    prepare_validation_result: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> QueryRunResult:
    """Run all iterations for a single query. Caller must wrap in try/except."""
    query_result = getattr(benchmark, f"q{q_id}")(run_config)
    q = query_result.frame

    print_query_plan(q_id, q, args, run_config, engine, print_plans=args.print_plans)
    plan = None
    if (args.explain or args.explain_logical) and engine is not None:
        from cudf_polars.experimental.explain import serialize_query

        plan = serialize_query(q, engine)

    casts = benchmark.EXPECTED_CASTS.get(q_id, [])
    if numeric_type == "decimal":
        casts.extend(benchmark.EXPECTED_CASTS_DECIMAL.get(q_id, []))
    if date_type == "timestamp":
        casts.extend(benchmark.EXPECTED_CASTS_TIMESTAMP.get(q_id, []))

    expected: pl.DataFrame | None = None
    if args.validate:
        if args.baseline == "cpu":
            expected = q.collect()
        elif args.baseline == "duckdb":
            duckdb_queries_cls = benchmark().duckdb_queries
            get_ddb = getattr(duckdb_queries_cls, f"q{q_id}")
            base_sql = get_ddb(run_config)
            expected = execute_duckdb_query(
                base_sql,
                run_config.dataset_path,
                query_set=duckdb_queries_cls.name,
                suffix=run_config.suffix,
                run_config=run_config,
            ).with_columns(*casts)
        else:
            raise ValueError(f"Invalid baseline: {args.baseline}")
    elif validation_files is not None:
        expected = pl.read_parquet(validation_files[q_id]).with_columns(*casts)
    else:
        expected = None

    if args.output_expected_directory is not None:
        assert expected is not None, (
            "Expected result must be computed before writing to disk."
        )
        expected_dir = Path(args.output_expected_directory)
        expected_dir.mkdir(parents=True, exist_ok=True)
        expected.write_parquet(expected_dir / f"q_{q_id:02d}.parquet")

    query_records: list[SuccessRecord | FailedRecord] = []
    iteration_failures: list[tuple[int, int]] = []
    validation_failed = False
    record: SuccessRecord | FailedRecord

    for i in range(args.iterations):
        if _HAS_STRUCTLOG and run_config.collect_traces:
            setup_logging(q_id, i)
            if engine is not None:
                engine._run(setup_logging, q_id, i)

        try:
            record = run_polars_query_iteration(
                q_id=q_id,
                iteration=i,
                q=q,
                run_config=run_config,
                args=args,
                engine=engine,
                expected=expected,
                query_result=query_result,
                prepare_validation_result=prepare_validation_result,
                result_casts=casts if casts else None,
            )
        except Exception:
            print(f"❌ query={q_id} iteration={i} failed!")
            print(traceback.format_exc())
            iteration_failures.append((q_id, i))
            record = FailedRecord(
                query=q_id,
                iteration=i,
                status="error",
                traceback=traceback.format_exc(),
            )

        else:
            if record.validation_result and record.validation_result.status == "Failed":
                validation_failed = True
                print(
                    f"❌ Query {q_id} failed validation!\n{record.validation_result.message}"
                )
                if record.validation_result.details:
                    pprint.pprint(record.validation_result.details)
            else:
                prefix = "✅ " if record.validation_result else ""
                print(
                    f"{prefix}Query {q_id} - Iteration {i} finished in {record.duration:0.4f}s",
                    flush=True,
                )

        query_records.append(record)

    return QueryRunResult(
        query_records=query_records,
        plan=plan,
        iteration_failures=iteration_failures,
        validation_failed=validation_failed,
    )


def _run_query_loop(
    benchmark: Any,
    args: argparse.Namespace,
    run_config: RunConfig,
    engine: StreamingEngine | None,
    numeric_type: str,
    date_type: str,
    validation_files: dict[int, Path] | None,
    prepare_validation_result: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> tuple[
    defaultdict[int, list[SuccessRecord | FailedRecord]],
    dict[int, Any],
    list[int],
    list[tuple[int, int]],
]:
    """Execute all queries in ``run_config`` and return accumulated results."""
    records: defaultdict[int, list[SuccessRecord | FailedRecord]] = defaultdict(list)
    plans: dict[int, SerializablePlan] = {}
    validation_failures: list[int] = []
    query_failures: list[tuple[int, int]] = []

    for q_id in run_config.queries:
        try:
            result = run_polars_query(
                q_id=q_id,
                benchmark=benchmark,
                run_config=run_config,
                args=args,
                engine=engine,
                numeric_type=numeric_type,
                date_type=date_type,
                validation_files=validation_files,
                prepare_validation_result=prepare_validation_result,
            )
        except Exception:
            print(f"❌ query={q_id} failed (setup or execution)!")
            print(traceback.format_exc())
            query_failures.append((q_id, -1))
            record = FailedRecord(
                query=q_id,
                iteration=-1,
                traceback=traceback.format_exc(),
            )
            result = QueryRunResult(
                query_records=[record],
                plan=None,
                iteration_failures=[],
                validation_failed=False,
            )

        records[q_id] = result.query_records
        if result.plan is not None:
            plans[q_id] = result.plan
        query_failures.extend(result.iteration_failures)
        if result.validation_failed:
            validation_failures.append(q_id)

    return records, plans, validation_failures, query_failures


def _finalize_benchmark_run(
    args: argparse.Namespace,
    run_config: RunConfig,
    validation_failures: list[int],
    query_failures: list[tuple[int, int]],
) -> None:
    """Summarize, serialize, and exit after a benchmark run."""
    if args.summarize:
        run_config.summarize()
    if args.validate and run_config.executor != "cpu":
        print("\nValidation Summary")
        print("==================")
        if validation_failures:
            print(
                f"{len(validation_failures)} queries failed validation: "
                f"{sorted(set(validation_failures))}"
            )
        else:
            print("✅ All validated queries passed.")
    args.output.write(json.dumps(run_config.serialize(engine=None)))
    args.output.write("\n")
    sys.exit(1 if (query_failures or validation_failures) else 0)


def run_polars_spmd(
    benchmark: Any,
    args: argparse.Namespace,
    run_config: Any,
    parquet_options: dict[str, Any],
    numeric_type: str,
    date_type: str,
    validation_files: dict[int, Path] | None,
) -> None:
    """Run benchmark queries using SPMD execution via the ``rrun`` launcher."""
    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    executor_options = get_executor_options(run_config, benchmark=benchmark)
    # "cluster" is reserved — SPMDEngine sets it
    executor_options.pop("cluster", None)
    engine_options = {
        **run_config.streaming_options.to_engine_options(),
        "parquet_options": parquet_options,
    }
    with SPMDEngine(
        rapidsmpf_options=run_config.streaming_options.to_rapidsmpf_options(),
        executor_options=executor_options,
        engine_options=engine_options,
    ) as engine:
        from cudf_polars.experimental.rapidsmpf.collectives.common import reserve_op_id
        from cudf_polars.experimental.rapidsmpf.frontend.spmd import (
            allgather_polars_dataframe,
        )

        def _allgather_result(df: pl.DataFrame) -> pl.DataFrame:
            with reserve_op_id() as op_id:
                return allgather_polars_dataframe(
                    engine=engine,
                    local_df=df,
                    op_id=op_id,
                )

        run_config = dataclasses.replace(run_config, n_workers=engine.nranks)
        records, plans, validation_failures, query_failures = _run_query_loop(
            benchmark,
            args,
            run_config,
            engine,
            numeric_type,
            date_type,
            validation_files,
            prepare_validation_result=_allgather_result,
        )
        if engine.rank > 0:
            sys.exit(1 if (query_failures or validation_failures) else 0)
        run_config = dataclasses.replace(run_config, records=dict(records), plans=plans)
        run_config = _consolidate_logs(
            run_config, engine=engine, gather_client_logs=False
        )
        _finalize_benchmark_run(args, run_config, validation_failures, query_failures)


def run_polars_ray(
    benchmark: Any,
    args: argparse.Namespace,
    run_config: Any,
    parquet_options: dict[str, Any],
    numeric_type: str,
    date_type: str,
    validation_files: dict[int, Path] | None,
) -> None:
    """Run benchmark queries using Ray actor-based distributed execution."""
    from cudf_polars.experimental.rapidsmpf.frontend.ray import RayEngine

    executor_options = get_executor_options(run_config, benchmark=benchmark)
    # "cluster" is reserved — RayEngine sets it
    executor_options.pop("cluster", None)
    engine_options: dict[str, Any] = {
        **run_config.streaming_options.to_engine_options(),
        "parquet_options": parquet_options,
    }
    ray_init_options: dict[str, Any] = {}
    if run_config.connect is not None:
        ray_init_options["address"] = run_config.connect
    if run_config.num_gpus is not None:
        ray_init_options["num_gpus"] = run_config.num_gpus

    with RayEngine(
        rapidsmpf_options=run_config.streaming_options.to_rapidsmpf_options(),
        executor_options=executor_options,
        engine_options=engine_options,
        ray_init_options=ray_init_options,
    ) as engine:
        run_config = dataclasses.replace(run_config, n_workers=engine.nranks)
        records, plans, validation_failures, query_failures = _run_query_loop(
            benchmark,
            args,
            run_config,
            engine,
            numeric_type,
            date_type,
            validation_files,
        )
        run_config = dataclasses.replace(run_config, records=dict(records), plans=plans)
        run_config = _consolidate_logs(run_config, engine=engine)

    _finalize_benchmark_run(args, run_config, validation_failures, query_failures)


def run_polars_dask(
    benchmark: Any,
    args: argparse.Namespace,
    run_config: Any,
    parquet_options: dict[str, Any],
    numeric_type: str,
    date_type: str,
    validation_files: dict[int, Path] | None,
) -> None:
    """Run benchmark queries using Dask distributed execution."""
    import distributed

    from cudf_polars.experimental.rapidsmpf.frontend.dask import DaskEngine

    executor_options = get_executor_options(run_config, benchmark=benchmark)
    # "cluster" is reserved — DaskEngine sets it
    executor_options.pop("cluster", None)
    engine_options: dict[str, Any] = {
        **run_config.streaming_options.to_engine_options(),
        "parquet_options": parquet_options,
    }
    dask_client = None
    if run_config.connect is not None:
        if Path(run_config.connect).is_file():
            dask_client = distributed.Client(scheduler_file=run_config.connect)
        else:
            dask_client = distributed.Client(address=run_config.connect)

    if run_config.num_gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(i) for i in range(run_config.num_gpus)
        )

    try:
        with DaskEngine(
            rapidsmpf_options=run_config.streaming_options.to_rapidsmpf_options(),
            executor_options=executor_options,
            engine_options=engine_options,
            dask_client=dask_client,
        ) as engine:
            run_config = dataclasses.replace(run_config, n_workers=engine.nranks)
            records, plans, validation_failures, query_failures = _run_query_loop(
                benchmark,
                args,
                run_config,
                engine,
                numeric_type,
                date_type,
                validation_files,
            )
            run_config = dataclasses.replace(
                run_config, records=dict(records), plans=plans
            )
            run_config = _consolidate_logs(run_config, engine)
    finally:
        if dask_client is not None:
            dask_client.close()
    _finalize_benchmark_run(args, run_config, validation_failures, query_failures)


def setup_logging(query_id: int, iteration: int) -> None:
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


def _consolidate_logs(
    run_config: RunConfig, engine: StreamingEngine, *, gather_client_logs: bool = True
) -> RunConfig:
    """Merge structlog traces from the local process and Dask workers into run_config."""
    if not (_HAS_STRUCTLOG and run_config.collect_traces):
        return run_config

    def gather_logs() -> str:
        logger = logging.getLogger()
        return logger.handlers[0].stream.getvalue()  # type: ignore[attr-defined]

    all_logs = "\n".join(engine._run(gather_logs))
    if gather_client_logs:
        all_logs += "\n" + gather_logs()

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

        new_records: list[SuccessRecord | FailedRecord] = []
        for rec, traces in zip(run_records, all_traces, strict=True):
            if rec.status == "success":
                new_records.append(dataclasses.replace(rec, traces=traces))
            else:
                new_records.append(rec)

        run_config.records[query_id] = new_records

    return run_config


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


def _make_duckdb_config(run_config: RunConfig | None) -> dict[str, Any]:
    """Build a DuckDB connection config dict from a RunConfig."""
    config: dict[str, Any] = {
        "threads": run_config.duckdb_threads
        if (run_config and run_config.duckdb_threads is not None)
        else os.cpu_count(),
    }
    if run_config and run_config.duckdb_memory_limit is not None:
        config["memory_limit"] = run_config.duckdb_memory_limit
    if run_config and run_config.duckdb_temp_dir is not None:
        config["temp_directory"] = run_config.duckdb_temp_dir
    return config


def print_duckdb_plan(
    q_id: int,
    sql: str,
    dataset_path: Path,
    suffix: str,
    query_set: str,
    args: argparse.Namespace,
    run_config: RunConfig | None = None,
) -> None:
    """Print DuckDB query plan using EXPLAIN."""
    if duckdb is None:
        raise ImportError(duckdb_err)

    if query_set == "pdsds":
        tbl_names = PDSDS_TABLE_NAMES
    else:
        tbl_names = PDSH_TABLE_NAMES

    with duckdb.connect(config=_make_duckdb_config(run_config)) as conn:
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
    run_config: RunConfig | None = None,
) -> pl.DataFrame:
    """Execute a query with DuckDB."""
    if duckdb is None:
        raise ImportError(duckdb_err)
    if query_set == "pdsds":
        tbl_names = PDSDS_TABLE_NAMES
    else:
        tbl_names = PDSH_TABLE_NAMES
    with duckdb.connect(config=_make_duckdb_config(run_config)) as conn:
        for name in tbl_names:
            pattern = (Path(dataset_path) / name).as_posix() + suffix
            conn.execute(
                f"CREATE OR REPLACE VIEW {name} AS "
                f"SELECT * FROM parquet_scan('{pattern}');"
            )
        return conn.execute(query).pl()


def run_duckdb(duckdb_queries_cls: Any, args: argparse.Namespace) -> None:
    """Run the benchmark with DuckDB."""
    vars(args).update({"query_set": duckdb_queries_cls.name})
    run_config = RunConfig.from_args(args)
    records: defaultdict[int, list[SuccessRecord | FailedRecord]] = defaultdict(list)

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
                run_config=run_config,
            )

        print(f"DuckDB Executing: {q_id}")
        records[q_id] = []

        for i in range(args.iterations):
            if run_config.io_mode == "cold":
                drop_file_page_cache_recursively(run_config.dataset_path)
            t0 = time.time()
            result = execute_duckdb_query(
                sql,
                run_config.dataset_path,
                suffix=run_config.suffix,
                query_set=duckdb_queries_cls.name,
                run_config=run_config,
            )
            t1 = time.time()
            record = SuccessRecord(query=q_id, iteration=i, duration=t1 - t0)
            if args.print_results:
                print(result)
            print(f"Query {q_id} - Iteration {i} finished in {record.duration:0.4f}s")
            records[q_id].append(record)
            if i == 0 and args.output_expected_directory is not None:
                expected_dir = Path(args.output_expected_directory)
                expected_dir.mkdir(parents=True, exist_ok=True)
                result.write_parquet(expected_dir / f"q_{q_id:02d}.parquet")

    run_config = dataclasses.replace(run_config, records=dict(records))
    if args.summarize:
        run_config.summarize()

    args.output.write(json.dumps(run_config.serialize(engine=None)))
    args.output.write("\n")


def list_validation_files(
    validate_directory: Path,
) -> dict[int, Path]:
    """List the validation files in the given directory."""
    validation_files: dict[int, Path] = {}
    for q_path in validate_directory.glob("q*.parquet"):
        q_id = int(q_path.stem.lstrip("q").lstrip("_"))
        validation_files[q_id] = q_path
    return validation_files


def check_input_data_type(
    run_config: RunConfig,
) -> tuple[Literal["decimal", "float"], Literal["date", "timestamp"]]:
    """
    Check the input data types for columns with variable data types.

    Our queries might be run on datasets that use different data types for different
    types of columns. Our validation supports:

    1. 'decimal' or 'float' for non-integer numeric columns (e.g. 'c_acctbal')
    2. 'date' or 'timestamp' for date type columns (e.g. 'o_orderdate')

    For PDS-H, this is determined by the ``c_acctbal`` column in the customer table.
    For PDS-DS, we use ``i_current_price`` from the item table.
    """
    if run_config.query_set == "pdsds":
        table, col = "item", "i_current_price"
    else:
        table, col = "customer", "c_acctbal"
    t = (
        get_data(run_config.dataset_path, table, run_config.suffix)
        .select(pl.col(col))
        .collect_schema()[col]
    )

    num_type: Literal["decimal", "float"]
    date_type: Literal["date", "timestamp"]
    if t.is_decimal():
        num_type = "decimal"
    else:
        num_type = "float"

    if run_config.query_set == "pdsds":
        date_type = "date"
    else:
        t = (
            get_data(run_config.dataset_path, "orders", run_config.suffix)
            .select(pl.col("o_orderdate"))
            .collect_schema()["o_orderdate"]
        )
        if t.to_python().__name__ == "date":
            date_type = "date"
        else:
            date_type = "timestamp"

    return num_type, date_type


def _query_type(num_queries: int) -> Any:
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


def build_parser(num_queries: int = 22) -> argparse.ArgumentParser:
    """Build the argument parser for PDS-H/PDS-DS benchmarks (new-frontend)."""
    from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions

    parser = argparse.ArgumentParser(
        prog="Cudf-Polars PDS-H/PDS-DS Benchmarks",
        description="Experimental streaming-executor benchmarks (SPMD / Ray / DuckDB).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "query",
        type=_query_type(num_queries),
        help=textwrap.dedent("""\
            Query to run. One of the following:
            - A single number (e.g. 11)
            - A comma-separated list of query numbers (e.g. 1,3,7)
            - A range of query numbers (e.g. 1-11,23-34)
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
        "--qualification",
        action="store_true",
        help="Use TPC-DS qualification parameters from specification Appendix B (PDS-DS only).",
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
        "--frontend",
        required=True,
        type=str,
        choices=["spmd", "ray", "dask", "duckdb"],
        help=textwrap.dedent("""\
            Execution frontend:
                - spmd   : SPMD execution via rrun launcher
                - ray    : Ray actor-based multi-GPU execution
                - dask   : Dask distributed multi-GPU execution
                - duckdb : DuckDB CPU execution"""),
    )
    parser.add_argument(
        "--connect",
        dest="connect",
        default=None,
        type=str,
        help=textwrap.dedent("""\
            Connect to an existing cluster instead of creating a local one.
            For --frontend dask: a TCP address (e.g. tcp://host:8786) or a
            scheduler file path. For --frontend ray: a Ray address
            (e.g. ray://host:10001 or "auto").
            Not supported with --frontend spmd."""),
    )
    parser.add_argument(
        "--num-gpus",
        dest="num_gpus",
        default=None,
        type=int,
        help="Number of GPUs for local cluster creation (--frontend ray/dask only). "
        "Cannot be used with --connect. Defaults to all visible GPUs.",
    )
    parser.add_argument(
        "--iterations",
        default=1,
        type=int,
        help="Number of times to run the same query.",
    )
    parser.add_argument(
        "--io-mode",
        dest="io_mode",
        default="lukewarm",
        choices=["cold", "lukewarm", "hot"],
        help=textwrap.dedent("""\
            Cache state control for each timed iteration:
                - cold     : Drop Linux page cache before each iteration (requires kvikio)
                - lukewarm : No cache manipulation; OS cache state unchanged (default)
                - hot      : One untimed warmup iteration to populate cache before measured runs"""),
    )
    parser.add_argument(
        "--collect-traces",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect data tracing cudf-polars execution.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Debug run.",
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
        help="Use C++ read_parquet nodes.",
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
        help="Print the query results.",
        default=True,
    )
    parser.add_argument(
        "--explain",
        action=argparse.BooleanOptionalAction,
        help="Print an outline of the physical plan.",
        default=False,
    )
    parser.add_argument(
        "--explain-logical",
        action=argparse.BooleanOptionalAction,
        help="Print an outline of the logical plan.",
        default=False,
    )
    parser.add_argument(
        "--print-plans",
        action=argparse.BooleanOptionalAction,
        help="Print the query plans.",
        default=True,
    )
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Validate the result against CPU execution. This will "
            "run the query with both GPU and baseline engine (CPU polars or DuckDB), collect the "
            "results in memory, and compare them using polars'. "
            "At larger scale factors, computing the expected result can be slow so "
            "--validate-directory should be used instead."
        ),
    )
    parser.add_argument(
        "--baseline",
        choices=["duckdb", "cpu"],
        default="duckdb",
        help="Which engine to use as the baseline for validation.",
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=None,
        help="Optional directory to write query results as parquet files.",
    )
    parser.add_argument(
        "--output-expected-directory",
        type=Path,
        default=None,
        help="Optional directory to write expected results as parquet files.",
    )
    parser.add_argument(
        "--validate-directory",
        type=Path,
        default=None,
        help=(
            "Validate the results against a directory with a pre-computed set of 'golden' results. "
            "The directory should contain one parquet file per query, named 'qDD.parquet', where DD is the "
            "zero-padded query number."
        ),
    )
    parser.add_argument(
        "--validation-abs-tol",
        type=float,
        default=0.01,
        help="Absolute tolerance for assert_frame_equal validation. Default: 0.01",
    )
    parser.add_argument(
        "--extra-info",
        type=json.loads,
        default={},
        help="Extra information to add to the output file (must be JSON-serializable).",
    )

    parser.add_argument(
        "--duckdb-threads",
        type=int,
        default=None,
        help="Number of threads for DuckDB to use. Defaults to os.cpu_count().",
    )
    parser.add_argument(
        "--duckdb-memory-limit",
        type=str,
        default=None,
        help="DuckDB memory limit (e.g. '500GB'). If unset, DuckDB uses its default.",
    )
    parser.add_argument(
        "--duckdb-temp-dir",
        type=str,
        default=None,
        help="Directory for DuckDB to spill temporary data to disk.",
    )

    StreamingOptions._add_cli_args(parser)

    # Trap legacy flags so we can emit clear errors.
    parser.add_argument(
        "--spill-device",
        dest="spill_device",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--blocksize",
        dest="blocksize",
        default=None,
        help=argparse.SUPPRESS,
    )

    return parser


def parse_args(
    args: Any = None,
    num_queries: int = 22,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    """Parse command line arguments."""
    if parser is None:
        parser = build_parser(num_queries)
    parsed_args = parser.parse_args(args)

    if parsed_args.spill_device is not None:
        parser.error(
            "--spill-device is not supported with --frontend; "
            "use --spill-device-limit instead, which takes a "
            'percentage, not a fraction (e.g. "80%").'
        )
    if parsed_args.blocksize is not None:
        parser.error(
            "--blocksize is not supported with --frontend; "
            "use --target-partition-size instead."
        )

    if parsed_args.validate_directory and parsed_args.validate:
        raise ValueError("Specify either --validate-directory or --validate, not both.")
    if (
        parsed_args.validate_directory is not None
        and not parsed_args.validate_directory.exists()
    ):
        raise FileNotFoundError(
            f"--validate-directory: {parsed_args.validate_directory} does not exist."
        )

    if (
        parsed_args.suffix
        and not parsed_args.suffix.startswith(".")
        and not parsed_args.suffix.startswith("/")
    ):
        parsed_args.suffix = f".{parsed_args.suffix}"

    return parsed_args


def run_polars(benchmark: Any, args: argparse.Namespace) -> None:
    """Run the queries using the given benchmark and frontend."""
    vars(args).update({"query_set": benchmark.name})
    run_config = RunConfig.from_args(args)

    if run_config.connect is not None and run_config.frontend == "spmd":
        raise ValueError("--connect is not supported with --frontend spmd.")

    if run_config.num_gpus is not None:
        if run_config.connect is not None:
            raise ValueError("--num-gpus cannot be used with --connect.")
        if run_config.frontend not in ("ray", "dask"):
            raise ValueError(
                "--num-gpus is only supported with --frontend ray or dask."
            )
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            raise ValueError(
                "--num-gpus cannot be used when CUDA_VISIBLE_DEVICES is already set. "
                "Unset CUDA_VISIBLE_DEVICES or use it directly to control GPU visibility."
            )

    parquet_options = {"use_rapidsmpf_native": run_config.native_parquet}
    validation_files = (
        list_validation_files(args.validate_directory)
        if args.validate_directory is not None
        else None
    )
    numeric_type, date_type = check_input_data_type(run_config)
    match args.frontend:
        case "spmd":
            run_polars_spmd(
                benchmark,
                args,
                run_config,
                parquet_options,
                numeric_type,
                date_type,
                validation_files,
            )
        case "ray":
            run_polars_ray(
                benchmark,
                args,
                run_config,
                parquet_options,
                numeric_type,
                date_type,
                validation_files,
            )
        case "dask":
            run_polars_dask(
                benchmark,
                args,
                run_config,
                parquet_options,
                numeric_type,
                date_type,
                validation_files,
            )
        case "duckdb":
            run_duckdb(benchmark, args)
        case _:
            raise ValueError(f"Unknown --frontend: {args.frontend!r}")
