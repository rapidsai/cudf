# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Config utilities."""

from __future__ import annotations

import dataclasses
import enum
import json
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from typing_extensions import Self

    import polars as pl


__all__ = ["ConfigOptions"]


# TODO: Use enum.StrEnum when we drop Python 3.10


class FallbackMode(str, enum.Enum):
    """
    How to handle errors when the GPU engine fails to execute a query.

    * ``FallbackMode.WARN`` : Emit a warning and fall back to the CPU engine.
    * ``FallbackMode.RAISE`` : Raise an exception.
    * ``FallbackMode.SILENT``: Silently fall back to the CPU engine.
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
    The method to use for shuffling data between workers.

    * ``ShuffleMethod.TASKS`` : Use the task-based shuffler.
    * ``ShuffleMethod.RAPIDSMFP`` : Use the rapidsmpf scheduler.
    """

    TASKS = "tasks"
    RAPIDSMFP = "rapidsmpf"


STREAMING_EXECUTOR_PARQUET_DEFAULTS = {
    "chunked": False,
}


@dataclasses.dataclass(frozen=True)
class ParquetOptions:
    """
    Configuration for the cudf-polars Parquet engine.

    Parameters
    ----------
    chunked
        Whether to use cudf's ``ChunkedParquetReader`` to read the parquet
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


@dataclasses.dataclass(frozen=True, eq=True)
class StreamingExecutor:
    """
    Configuration for the cudf-polars streaming executor.

    Parameters
    ----------
    scheduler
        The scheduler to use for the streaming executor. ``Scheduler.SYNCHRONOUS``
        by default.

        Note taht the "distributed" requires a Dask cluster to be running.

    fallback_mode
        How to handle errors when the GPU engine fails to execute a query.
        ``FallbackMode.WARN`` by default.
    max_rows_per_partition
        The maximum number of rows to process per partition. 1,000,000 by default.
        When the number of rows exceeds this value, the query will be split into
        multiple partitions and executed in parallel.
    cardinality_factor
        A dictionary mapping column names to a float strictly greater than 0 and
        less than or equal to 1. Each factor estimates the fractional number of
        unique values in the column. By default, a value of ``1.0`` is used
        for any column not found in ``cardinality_factor``.
    parquet_blocksize
        Controls how large parquet files are split into multiple partitions.
        Files larger than ``parquet_blocksize`` bytes are split into multiple
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
        The method to use for shuffling data between workers. ``ShuffleMethod.TASKS``
        by default.
    """

    name: Literal["streaming"] = dataclasses.field(default="streaming", init=False)
    scheduler: Scheduler = Scheduler.SYNCHRONOUS
    fallback_mode: FallbackMode = FallbackMode.WARN
    max_rows_per_partition: int = 1_000_000
    cardinality_factor: dict[str, float] = dataclasses.field(default_factory=dict)
    parquet_blocksize: int = 1_000_000_000  # why isn't this a ParquetOption?
    groupby_n_ary: int = 32
    broadcast_join_limit: int = 4
    shuffle_method: ShuffleMethod | None = None

    def __post_init__(self) -> None:
        if self.scheduler == "synchronous" and self.shuffle_method == "rapidsmpf":
            raise ValueError(
                "rapidsmpf shuffle method is not supported for synchronous scheduler"
            )

        # frozen dataclass, so use object.__setattr__
        object.__setattr__(self, "fallback_mode", FallbackMode(self.fallback_mode))
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
        if not isinstance(self.parquet_blocksize, int):
            raise TypeError("parquet_blocksize must be an int")
        if not isinstance(self.groupby_n_ary, int):
            raise TypeError("groupby_n_ary must be an int")
        if not isinstance(self.broadcast_join_limit, int):
            raise TypeError("broadcast_join_limit must be an int")

    def __hash__(self) -> int:
        # cardinatlity factory, a dict, isn't natively hashable. We'll dump it
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
    shuffle_method
        The method to use for shuffling data between workers. Currently only
        ``ShuffleMethod.TASKS`` is supported for the in-memory executor.
    broadcast_join_limit
        The maximum number of partitions to allow for the smaller table in
        a broadcast join.
    """

    name: Literal["in-memory"] = dataclasses.field(default="in-memory", init=False)
    scheduler: Literal["synchronous"] = "synchronous"
    shuffle_method: Literal["tasks"] | None = None
    broadcast_join_limit: int = 32

    def __post_init__(self) -> None:
        if self.scheduler != "synchronous":
            raise ValueError(
                "'synchronous' is the only valid scheduler for the in-memory executor"
            )
        if self.shuffle_method is not None and self.shuffle_method != "tasks":
            raise ValueError(
                "'tasks' is the only valid shuffle method for the in-memory executor"
            )

        if not isinstance(self.broadcast_join_limit, int):
            raise TypeError("broadcast_join_limit must be an int")


@dataclasses.dataclass(frozen=True, eq=True)
class ConfigOptions:
    """
    Configuration for the polars GPUEngine.

    Parameters
    ----------
    raise_on_fail
        Whether to raise an exception when the GPU engine fails to execute a
        query. ``False`` by default.
    parquet_options
        Options controlling parquet file reading and writing. See
        :class:`ParquetOptions` for more.
    executor
        The executor to use for the GPU engine. See :class:`StreamingExecutor`
        and :class:`InMemoryExecutor` for more.
    device
        The device to use for the GPU engine.
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
                user_parquet_options = {
                    **STREAMING_EXECUTOR_PARQUET_DEFAULTS,
                    **user_parquet_options,
                }

            case _:
                raise ValueError(f"Unsupported executor: {user_executor}")

        return cls(
            raise_on_fail=user_raise_on_fail,
            parquet_options=ParquetOptions(**user_parquet_options),
            executor=executor,
            device=engine.device,
        )
