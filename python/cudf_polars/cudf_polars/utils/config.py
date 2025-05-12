# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Config utilities."""

from __future__ import annotations

import dataclasses
import enum
import json
import os
import warnings
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from typing_extensions import Self

    import polars as pl


__all__ = ["ConfigOptions"]


# TODO: Use enum.StrEnum when we drop Python 3.10


class StreamingFallbackMode(str, enum.Enum):
    """
    How the streaming executor handles operations that don't support multiple partitions.

    Upon encountering an unsupported operation, the streaming executor will fall
    back to using a single partition, which might use a large amount of memory.

    * ``StreamingFallbackMode.WARN`` : Emit a warning and fall back to a single partition.
    * ``StreamingFallbackMode.SILENT``: Silently fall back to a single partition.
    * ``StreamingFallbackMode.RAISE`` : Raise an exception.
    """

    WARN = "warn"
    RAISE = "raise"
    SILENT = "silent"


class Scheduler(str, enum.Enum):
    """
    The scheduler to use for the streaming executor.

    * ``Scheduler.SYNCHRONOUS`` : Use the synchronous scheduler.
    * ``Scheduler.DISTRIBUTED`` : Use the distributed scheduler.
    """

    SYNCHRONOUS = "synchronous"
    DISTRIBUTED = "distributed"


class ShuffleMethod(str, enum.Enum):
    """
    The method to use for shuffling data between workers with the streaming executor.

    * ``ShuffleMethod.TASKS`` : Use the task-based shuffler.
    * ``ShuffleMethod.RAPIDSMPF`` : Use the rapidsmpf scheduler.

    In :class:`StreamingExecutor`, the default of ``None`` will attempt to use
    ``ShuffleMethod.RAPIDSMPF``, but will fall back to ``ShuffleMethod.TASKS``
    if rapidsmpf is not installed.
    """

    TASKS = "tasks"
    RAPIDSMPF = "rapidsmpf"


@dataclasses.dataclass(frozen=True)
class ParquetOptions:
    """
    Configuration for the cudf-polars Parquet engine.

    Parameters
    ----------
    chunked
        Whether to use libcudf's ``ChunkedParquetReader`` to read the parquet
        dataset in chunks. This is useful when reading very large parquet
        files.
    chunk_read_limit
        Limit on total number of bytes to be returned per read, or 0 if
        there is no limit.
    pass_read_limit
        Limit on the amount of memory used for reading and decompressing data
        or 0 if there is no limit.
    """

    chunked: bool = True
    chunk_read_limit: int = 0
    pass_read_limit: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.chunked, bool):
            raise TypeError("chunked must be a bool")
        if not isinstance(self.chunk_read_limit, int):
            raise TypeError("chunk_read_limit must be an int")
        if not isinstance(self.pass_read_limit, int):
            raise TypeError("pass_read_limit must be an int")


def default_blocksize(scheduler: str) -> int:
    """Return the default blocksize."""
    try:
        # Use PyNVML to find the worker device size.
        import pynvml

        pynvml.nvmlInit()
        index = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        if index and not index.isnumeric():  # pragma: no cover
            # This means device_index is UUID.
            # This works for both MIG and non-MIG device UUIDs.
            handle = pynvml.nvmlDeviceGetHandleByUUID(str.encode(index))
            if pynvml.nvmlDeviceIsMigDeviceHandle(handle):
                # Additionally get parent device handle
                # if the device itself is a MIG instance
                handle = pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle)
        else:
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(index))

        device_size = pynvml.nvmlDeviceGetMemoryInfo(handle).total

    except (ImportError, ValueError, pynvml.NVMLError) as err:  # pragma: no cover
        # Fall back to a conservative 12GiB default
        warnings.warn(
            "Failed to query the device size with NVML. Please "
            "set 'target_partition_size' to a literal byte size to "
            f"silence this warning. Original error: {err}",
            stacklevel=1,
        )
        device_size = 12 * 1024**3

    if scheduler == "distributed":
        # Distributed execution requires a conservative
        # blocksize for now.
        blocksize = int(device_size * 0.025)
    else:
        # Single-GPU execution can lean on UVM to
        # support a much larger blocksize.
        blocksize = int(device_size * 0.0625)

    return max(blocksize, 256_000_000)


@dataclasses.dataclass(frozen=True, eq=True)
class StreamingExecutor:
    """
    Configuration for the cudf-polars streaming executor.

    Parameters
    ----------
    scheduler
        The scheduler to use for the streaming executor. ``Scheduler.SYNCHRONOUS``
        by default.

        Note ``scheduler="distributed"`` requires a Dask cluster to be running.
    fallback_mode
        How to handle errors when the GPU engine fails to execute a query.
        ``StreamingFallbackMode.WARN`` by default.
    max_rows_per_partition
        The maximum number of rows to process per partition. 1_000_000 by default.
        When the number of rows exceeds this value, the query will be split into
        multiple partitions and executed in parallel.
    cardinality_factor
        A dictionary mapping column names to floats between 0 and 1 (inclusive
        on the right).

        Each factor estimates the fractional number of unique values in the
        column. By default, ``1.0`` is used for any column not included in
        ``cardinality_factor``.
    target_partition_size
        Target partition size for IO tasks. This configuration currently
        controls how large parquet files are split into multiple partitions.
        Files larger than ``target_partition_size`` bytes are split into multiple
        partitions.
    groupby_n_ary
        The factor by which the number of partitions is decreased when performing
        a groupby on a partitioned column. For example, if a column has 64 partitions,
        it will first be reduced to ``ceil(64 / 32) = 2`` partitions.

        This is useful when the absolute number of partitions is large.
    broadcast_join_limit
        The maximum number of partitions to allow for the smaller table in
        a broadcast join.
    shuffle_method
        The method to use for shuffling data between workers. ``None``
        by default, which will use 'rapidsmpf' if installed and fall back to
        'tasks' if not.
    rapidsmpf_spill
        Whether to wrap task arguments and output in objects that are
        spillable by 'rapidsmpf'.
    """

    name: Literal["streaming"] = dataclasses.field(default="streaming", init=False)
    scheduler: Scheduler = Scheduler.SYNCHRONOUS
    fallback_mode: StreamingFallbackMode = StreamingFallbackMode.WARN
    max_rows_per_partition: int = 1_000_000
    cardinality_factor: dict[str, float] = dataclasses.field(default_factory=dict)
    target_partition_size: int = 0
    groupby_n_ary: int = 32
    broadcast_join_limit: int = 0
    shuffle_method: ShuffleMethod | None = None
    rapidsmpf_spill: bool = False

    def __post_init__(self) -> None:
        if self.scheduler == "synchronous" and self.shuffle_method == "rapidsmpf":
            raise ValueError(
                "rapidsmpf shuffle method is not supported for synchronous scheduler"
            )

        # frozen dataclass, so use object.__setattr__
        object.__setattr__(
            self, "fallback_mode", StreamingFallbackMode(self.fallback_mode)
        )
        if self.target_partition_size == 0:
            object.__setattr__(
                self, "target_partition_size", default_blocksize(self.scheduler)
            )
        if self.broadcast_join_limit == 0:
            object.__setattr__(
                self,
                "broadcast_join_limit",
                # Usually better to avoid shuffling for single gpu
                2 if self.scheduler == "distributed" else 32,
            )
        object.__setattr__(self, "scheduler", Scheduler(self.scheduler))
        if self.shuffle_method is not None:
            object.__setattr__(
                self, "shuffle_method", ShuffleMethod(self.shuffle_method)
            )

        # Type / value check everything else
        if not isinstance(self.max_rows_per_partition, int):
            raise TypeError("max_rows_per_partition must be an int")
        if not isinstance(self.cardinality_factor, dict):
            raise TypeError("cardinality_factor must be a dict of column name to float")
        if not isinstance(self.target_partition_size, int):
            raise TypeError("target_partition_size must be an int")
        if not isinstance(self.groupby_n_ary, int):
            raise TypeError("groupby_n_ary must be an int")
        if not isinstance(self.broadcast_join_limit, int):
            raise TypeError("broadcast_join_limit must be an int")
        if not isinstance(self.rapidsmpf_spill, bool):
            raise TypeError("rapidsmpf_spill must be bool")

    def __hash__(self) -> int:
        # cardinality factory, a dict, isn't natively hashable. We'll dump it
        # to json and hash that.
        d = dataclasses.asdict(self)
        d["cardinality_factor"] = json.dumps(d["cardinality_factor"])
        return hash(tuple(sorted(d.items())))


@dataclasses.dataclass(frozen=True, eq=True)
class InMemoryExecutor:
    """
    Configuration for the cudf-polars in-memory executor.

    Parameters
    ----------
    scheduler:
        The scheduler to use for the in-memory executor. Currently
        only ``Scheduler.SYNCHRONOUS`` is supported for the in-memory executor.
    """

    name: Literal["in-memory"] = dataclasses.field(default="in-memory", init=False)


@dataclasses.dataclass(frozen=True, eq=True)
class ConfigOptions:
    """
    Configuration for the polars GPUEngine.

    Parameters
    ----------
    raise_on_fail
        Whether to raise an exception when the GPU engine cannot execute a
        query. ``False`` by default.
    parquet_options
        Options controlling parquet file reading and writing. See
        :class:`ParquetOptions` for more.
    executor
        The executor to use for the GPU engine. See :class:`StreamingExecutor`
        and :class:`InMemoryExecutor` for more.
    device
        The GPU used to run the query. If not provided, the
        query uses the current CUDA device.
    """

    raise_on_fail: bool = False
    parquet_options: ParquetOptions = dataclasses.field(default_factory=ParquetOptions)
    executor: StreamingExecutor | InMemoryExecutor = dataclasses.field(
        default_factory=InMemoryExecutor
    )
    device: int | None = None

    @classmethod
    def from_polars_engine(cls, engine: pl.GPUEngine) -> Self:
        """
        Create a `ConfigOptions` object from a `pl.GPUEngine` object.

        This creates our internal, typed, configuration object from the
        user-provided `polars.GPUEngine` object.
        """
        # these are the valid top-level keys in the engine.config that
        # the user passes as **kwargs to GPUEngine.
        valid_options = {
            "executor",
            "executor_options",
            "parquet_options",
            "raise_on_fail",
        }

        extra_options = set(engine.config.keys()) - valid_options
        if extra_options:
            raise TypeError(f"Unsupported executor_options: {extra_options}")

        user_executor = engine.config.get("executor", "in-memory")
        if user_executor is None:
            user_executor = "in-memory"
        user_executor_options = engine.config.get("executor_options", {})
        user_parquet_options = engine.config.get("parquet_options", {})
        user_raise_on_fail = engine.config.get("raise_on_fail", False)

        # These are user-provided options, so we need to actually validate
        # them.

        if user_executor not in {"in-memory", "streaming"}:
            raise ValueError(f"Unknown executor '{user_executor}'")

        if not isinstance(user_raise_on_fail, bool):
            raise TypeError("GPUEngine option 'raise_on_fail' must be a boolean.")

        executor: InMemoryExecutor | StreamingExecutor

        match user_executor:
            case "in-memory":
                executor = InMemoryExecutor(**user_executor_options)
            case "streaming":
                executor = StreamingExecutor(**user_executor_options)
                # Update with the streaming defaults, but user options take precedence.

            case _:  # pragma: no cover; Unreachable
                raise ValueError(f"Unsupported executor: {user_executor}")

        return cls(
            raise_on_fail=user_raise_on_fail,
            parquet_options=ParquetOptions(**user_parquet_options),
            executor=executor,
            device=engine.device,
        )
