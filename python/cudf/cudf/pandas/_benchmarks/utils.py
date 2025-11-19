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
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

import nvtx

import cudf.pandas
from cudf.pandas.module_accelerator import disable_module_accelerator

cudf.pandas.install()

import pandas as pd  # noqa: E402

try:
    import pynvml
except ImportError:
    pynvml = None

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path


ExecutorType = Literal["in-memory", "cpu"]


@dataclasses.dataclass
class Record:
    """Results for a single run of a single PDS-H query."""

    query: int
    duration: float
    shuffle_stats: None = None


@dataclasses.dataclass
class VersionInfo:
    """Information about the commit of the software used to run the query."""

    version: str
    commit: str


@dataclasses.dataclass
class PackageVersions:
    """Information about the versions of the software used to run the query."""

    cudf: str | VersionInfo
    pandas: str
    python: str

    @classmethod
    def collect(cls) -> PackageVersions:
        """Collect the versions of the software used to run the query."""
        packages = [
            "cudf",
            "pandas",
        ]
        versions: dict[str, str | VersionInfo | None] = {}
        for name in packages:
            try:
                package = importlib.import_module(name)
            except (AttributeError, ImportError):
                versions[name] = None
            else:
                if name == "cudf":
                    versions[name] = VersionInfo(
                        version=package.__version__,
                        commit=package.__git_commit__,
                    )
                else:
                    versions[name] = package.__version__

        versions["python"] = ".".join(str(v) for v in sys.version_info[:3])
        # we manually ensure that only cudf and pandas have a VersionInfo
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
            gpus = [
                GPUInfo.from_index(i)
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
        else:
            # No GPUs -- probably running in CPU mode
            gpus = []
        return cls(gpus=gpus)


def _infer_scale_factor(
    name: str, path: str | Path, suffix: str
) -> int | float:
    if "pdsh" in name:
        supplier = get_data(path, "supplier", suffix)
        num_rows = len(supplier)
        return num_rows / 10_000

    elif "pdsds" in name:
        # TODO: Keep a map of SF-row_count because of nonlinear scaling
        # See: https://www.tpc.org/TPC_Documents_Current_Versions/pdf/TPC-DS_v4.0.0.pdf pg.46
        customer = get_data(path, "promotion", suffix)
        num_rows = len(customer)
        return num_rows / 300

    else:
        raise ValueError(f"Invalid benchmark script name: '{name}'.")


@dataclasses.dataclass(kw_only=True)
class RunConfig:
    """Results for a PDS-H or PDS-DS query run."""

    queries: list[int]
    suffix: str
    executor: ExecutorType
    versions: PackageVersions = dataclasses.field(
        default_factory=PackageVersions.collect
    )
    records: dict[int, list[Record]] = dataclasses.field(default_factory=dict)
    dataset_path: Path
    scale_factor: int | float
    iterations: int
    timestamp: str = dataclasses.field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    hardware: HardwareInfo = dataclasses.field(
        default_factory=HardwareInfo.collect
    )
    query_set: str

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RunConfig:
        """Create a RunConfig from command line arguments."""
        executor: ExecutorType = args.executor

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

        if "pdsh" in name and args.scale is not None:
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
            dataset_path=path,
            scale_factor=scale_factor,
            iterations=args.iterations,
            suffix=args.suffix,
            query_set=args.query_set,
        )

    def serialize(self) -> dict:
        """Serialize the run config to a dictionary."""
        return dataclasses.asdict(self)

    def summarize(self) -> None:
        """Print a summary of the results."""
        print("Iteration Summary")  # noqa: T201
        print("=======================================")  # noqa: T201

        for query, records in self.records.items():
            print(f"query: {query}")  # noqa: T201
            print(f"path: {self.dataset_path}")  # noqa: T201
            print(f"scale_factor: {self.scale_factor}")  # noqa: T201
            print(f"executor: {self.executor}")  # noqa: T201
            if len(records) > 0:
                print(f"iterations: {self.iterations}")  # noqa: T201
                print("---------------------------------------")  # noqa: T201
                print(  # noqa: T201
                    f"min time : {min(record.duration for record in records):0.4f}"
                )
                print(  # noqa: T201
                    f"max time : {max(record.duration for record in records):0.4f}"
                )
                print(  # noqa: T201
                    f"mean time: {statistics.mean(record.duration for record in records):0.4f}"
                )
                print("=======================================")  # noqa: T201
        total_mean_time = sum(
            statistics.mean(record.duration for record in records)
            for records in self.records.values()
            if records
        )
        print(  # noqa: T201
            f"Total mean time across all queries: {total_mean_time:.4f} seconds"
        )


def get_data(
    path: str | Path, table_name: str, suffix: str = ""
) -> pd.DataFrame:
    """Get table from dataset."""
    return pd.read_parquet(f"{path}/{table_name}{suffix}")


def execute_query(
    q_id: int,
    i: int,
    q: Callable[[RunConfig], pd.DataFrame],
    run_config: RunConfig,
) -> pd.DataFrame:
    """Execute a query with NVTX annotation."""
    with nvtx.annotate(
        message=f"Query {q_id} - Iteration {i}",
        domain="cudf.pandas",
        color="green",
    ):
        if run_config.executor == "cpu":
            with disable_module_accelerator():
                assert not cudf.pandas.LOADED
                return q(run_config)
        assert cudf.pandas.LOADED
        return q(run_config)


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
        prog="cudf.pandas PDS-H Benchmarks",
        description="cudf.pandas benchmark runner.",
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
        default="in-memory",
        type=str,
        choices=["in-memory", "cpu"],
        help=textwrap.dedent("""\
            Query executor backend:
                - in-memory : Use cudf.pandas
                - cpu       : Use pandas"""),
    )
    parser.add_argument(
        "--iterations",
        default=1,
        type=int,
        help="Number of times to run the same query.",
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


def run_pandas(
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

    records: defaultdict[int, list[Record]] = defaultdict(list)

    for q_id in run_config.queries:
        try:
            q = getattr(benchmark, f"q{q_id}")
        except AttributeError as err:
            raise NotImplementedError(
                f"Query {q_id} not implemented."
            ) from err

        records[q_id] = []

        for i in range(args.iterations):
            t0 = time.monotonic()

            try:
                result = execute_query(q_id, i, q, run_config)
            except Exception:
                print(f"❌ query={q_id} iteration={i} failed!")  # noqa: T201
                print(traceback.format_exc())  # noqa: T201
                query_failures.append((q_id, i))
                continue

            t1 = time.monotonic()
            if args.validate and run_config.executor != "cpu":
                cpu_run_config = dataclasses.replace(
                    run_config, executor="cpu"
                )
                cpu_result = execute_query(q_id, i, q, cpu_run_config)
                try:
                    with disable_module_accelerator():
                        pd.testing.assert_frame_equal(
                            result._fsproxy_slow, cpu_result
                        )
                    print(f"✅ Query {q_id} passed validation!")  # noqa: T201
                except AssertionError as e:
                    validation_failures.append(q_id)
                    print(f"❌ Query {q_id} failed validation!\n{e}")  # noqa: T201

            record = Record(query=q_id, duration=t1 - t0, shuffle_stats=None)
            if args.print_results:
                print(result)  # noqa: T201

            print(  # noqa: T201
                f"Query {q_id} - Iteration {i} finished in {record.duration:0.4f}s"
            )
            records[q_id].append(record)

    run_config = dataclasses.replace(run_config, records=dict(records))

    if args.summarize:
        run_config.summarize()

    if args.validate and run_config.executor != "cpu":
        print("\nValidation Summary")  # noqa: T201
        print("==================")  # noqa: T201
        if validation_failures:
            print(  # noqa: T201
                f"{len(validation_failures)} queries failed validation: {sorted(set(validation_failures))}"
            )
        else:
            print("All validated queries passed.")  # noqa: T201

    args.output.write(json.dumps(run_config.serialize()))
    args.output.write("\n")

    if query_failures or validation_failures:
        sys.exit(1)
