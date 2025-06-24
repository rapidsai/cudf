# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions/classes for running the PDS-H and TPC-DS (inspired) benchmarks."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import os
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

import polars as pl

try:
    import pynvml
except ImportError:
    pynvml = None

if TYPE_CHECKING:
    import pathlib
    from collections.abc import Sequence

import textwrap


@dataclasses.dataclass
class Record:
    """Results for a single run of a single PDS-H query."""

    query: int
    duration: float


@dataclasses.dataclass
class PackageVersions:
    """Information about the versions of the software used to run the query."""

    cudf_polars: str
    polars: str
    python: str
    rapidsmpf: str | None

    @classmethod
    def collect(cls) -> PackageVersions:
        """Collect the versions of the software used to run the query."""
        packages = [
            "cudf_polars",
            "polars",
            "rapidsmpf",
        ]
        versions = {}
        for name in packages:
            try:
                package = importlib.import_module(name)
                versions[name] = package.__version__
            except (AttributeError, ImportError):  # noqa: PERF203
                versions[name] = None
        versions["python"] = ".".join(str(v) for v in sys.version_info[:3])
        return cls(**versions)


@dataclasses.dataclass
class GPUInfo:
    """Information about a specific GPU."""

    name: str
    index: int
    free_memory: int
    used_memory: int
    total_memory: int

    @classmethod
    def from_index(cls, index: int) -> GPUInfo:
        """Create a GPUInfo from an index."""
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return cls(
            name=pynvml.nvmlDeviceGetName(handle),
            index=index,
            free_memory=memory.free,
            used_memory=memory.used,
            total_memory=memory.total,
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


def _infer_scale_factor(path: str | pathlib.Path, suffix: str) -> int | float:
    # Use "supplier" table to infer the scale-factor
    supplier = get_data(path, "supplier", suffix)
    num_rows = supplier.select(pl.len()).collect().item(0, 0)
    return num_rows / 10_000


@dataclasses.dataclass(kw_only=True)
class RunConfig:
    """Results for a PDS-H query run."""

    queries: list[int]
    suffix: str
    executor: str
    scheduler: str
    n_workers: int
    versions: PackageVersions = dataclasses.field(
        default_factory=PackageVersions.collect
    )
    records: dict[int, list[Record]] = dataclasses.field(default_factory=dict)
    dataset_path: pathlib.Path
    scale_factor: int | float
    shuffle: str | None = None
    broadcast_join_limit: int | None = None
    blocksize: int | None = None
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
        executor = args.executor
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
        )

    def serialize(self) -> dict:
        """Serialize the run config to a dictionary."""
        return dataclasses.asdict(self)

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
                print(f"min time : {min([record.duration for record in records]):0.4f}")
                print(f"max time : {max(record.duration for record in records):0.4f}")
                print(
                    f"mean time: {np.mean([record.duration for record in records]):0.4f}"
                )
                print("=======================================")


def get_data(
    path: str | pathlib.Path, table_name: str, suffix: str = ""
) -> pl.LazyFrame:
    """Get table from dataset."""
    return pl.scan_parquet(f"{path}/{table_name}{suffix}")


def _query_type(query: int | str) -> list[int]:
    if isinstance(query, int):
        return [query]
    elif query == "all":
        return list(range(1, 23))
    else:
        return [int(q) for q in query.split(",")]


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="Cudf-Polars PDS-H Benchmarks",
        description="Experimental streaming-executor benchmarks.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "query",
        type=_query_type,
        help=textwrap.dedent("""\
            Query to run. One of the following:
            - A single number (e.g., 11)
            - A comma-separated list of query numbers (e.g., 1,3,7)
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
                - synchronous : Run locally single-process
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
        help="Approx. partition size.",
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
        choices=["ucx", "ucxx"],
        help="Communication protocol to use for Dask: ucx (UCX-Py) or ucxx)",
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
    return parser.parse_args(args)
