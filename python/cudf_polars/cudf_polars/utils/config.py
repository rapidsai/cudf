# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration utilities for the cudf-polars engine.

Most users will not construct these objects directly. Instead, you'll pass
keyword arguments to :class:`~polars.lazyframe.engine_config.GPUEngine`. The
majority of the options are passed as `**kwargs` and collected into the
configuration described below:

.. code-block:: python

   >>> import polars as pl
   >>> engine = pl.GPUEngine(
   ...     executor="streaming",
   ...     executor_options={"fallback_mode": "raise"}
   ... )

"""

from __future__ import annotations

import dataclasses
import enum
import functools
import importlib.util
import json
import os
import warnings
from typing import TYPE_CHECKING, Literal, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self

    import polars.lazyframe.engine_config


__all__ = [
    "ConfigOptions",
    "InMemoryExecutor",
    "ParquetOptions",
    "Scheduler",
    "ShuffleMethod",
    "StreamingExecutor",
    "StreamingFallbackMode",
]


def _env_get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (ValueError, TypeError):  # pragma: no cover
        return default  # pragma: no cover


def get_total_device_memory() -> int | None:
    """Return the total memory of the current device."""
    import pynvml

    try:
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

        return pynvml.nvmlDeviceGetMemoryInfo(handle).total

    except pynvml.NVMLError_NotSupported:  # pragma: no cover
        # System doesn't have proper "GPU memory".
        return None


@functools.cache
def rapidsmpf_available() -> bool:  # pragma: no cover
    """Query whether rapidsmpf is available as a shuffle method."""
    try:
        return importlib.util.find_spec("rapidsmpf.integrations.dask") is not None
    except (ImportError, ValueError):
        return False


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

    * ``Scheduler.SYNCHRONOUS`` : A zero-dependency, synchronous,
      single-threaded scheduler.
    * ``Scheduler.DISTRIBUTED`` : A Dask-based distributed scheduler.
      Using this scheduler requires an active Dask cluster.
    """

    SYNCHRONOUS = "synchronous"
    DISTRIBUTED = "distributed"


class ShuffleMethod(str, enum.Enum):
    """
    The method to use for shuffling data between workers with the streaming executor.

    * ``ShuffleMethod.TASKS`` : Use the task-based shuffler.
    * ``ShuffleMethod.RAPIDSMPF`` : Use the rapidsmpf scheduler.

    With :class:`cudf_polars.utils.config.StreamingExecutor`, the default of ``None`` will attempt to use
    ``ShuffleMethod.RAPIDSMPF``, but will fall back to ``ShuffleMethod.TASKS``
    if rapidsmpf is not installed.
    """

    TASKS = "tasks"
    RAPIDSMPF = "rapidsmpf"


T = TypeVar("T")


def _make_default_factory(
    key: str, converter: Callable[[str], T], *, default: T
) -> Callable[[], T]:
    def default_factory() -> T:
        v = os.environ.get(key)
        if v is None:
            return default
        return converter(v)

    return default_factory


def _bool_converter(v: str) -> bool:
    lowered = v.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    elif lowered in {"0", "false", "no", "n"}:
        return False
    else:
        raise ValueError(f"Invalid boolean value: '{v}'")


@dataclasses.dataclass(frozen=True)
class ParquetOptions:
    """
    Configuration for the cudf-polars Parquet engine.

    These options can be configured via environment variables
    with the prefix ``CUDF_POLARS__PARQUET_OPTIONS__``.

    Parameters
    ----------
    chunked
        Whether to use libcudf's ``ChunkedParquetReader`` or ``ChunkedParquetWriter``
        to read/write the parquet dataset in chunks. This is useful when reading/writing
        very large parquet files.
    n_output_chunks
        Split the dataframe in ``n_output_chunks`` when using libcudf's ``ChunkedParquetWriter``.
    chunk_read_limit
        Limit on total number of bytes to be returned per read, or 0 if
        there is no limit.
    pass_read_limit
        Limit on the amount of memory used for reading and decompressing data
        or 0 if there is no limit.
    max_footer_samples
        Maximum number of file footers to sample for metadata. This
        option is currently used by the streaming executor to gather
        datasource statistics before generating a physical plan. Set to
        0 to avoid metadata sampling. Default is 3.
    max_row_group_samples
        Maximum number of row-groups to sample for unique-value statistics.
        This option may be used by the streaming executor to optimize
        the physical plan. Default is 1.

        Set to 0 to avoid row-group sampling. Note that row-group sampling
        will also be skipped if ``max_footer_samples`` is 0.
    """

    _env_prefix = "CUDF_POLARS__PARQUET_OPTIONS"

    chunked: bool = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__CHUNKED", _bool_converter, default=True
        )
    )
    n_output_chunks: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__N_OUTPUT_CHUNKS", int, default=1
        )
    )
    chunk_read_limit: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__CHUNK_READ_LIMIT", int, default=0
        )
    )
    pass_read_limit: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__PASS_READ_LIMIT", int, default=0
        )
    )
    max_footer_samples: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__MAX_FOOTER_SAMPLES", int, default=3
        )
    )
    max_row_group_samples: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__MAX_ROW_GROUP_SAMPLES", int, default=1
        )
    )

    def __post_init__(self) -> None:  # noqa: D105
        if not isinstance(self.chunked, bool):
            raise TypeError("chunked must be a bool")
        if not isinstance(self.n_output_chunks, int):
            raise TypeError("n_output_chunks must be an int")
        if not isinstance(self.chunk_read_limit, int):
            raise TypeError("chunk_read_limit must be an int")
        if not isinstance(self.pass_read_limit, int):
            raise TypeError("pass_read_limit must be an int")
        if not isinstance(self.max_footer_samples, int):
            raise TypeError("max_footer_samples must be an int")
        if not isinstance(self.max_row_group_samples, int):
            raise TypeError("max_row_group_samples must be an int")


def default_blocksize(scheduler: str) -> int:
    """Return the default blocksize."""
    device_size = get_total_device_memory()
    if device_size is None:  # pragma: no cover
        # System doesn't have proper "GPU memory".
        # Fall back to a conservative 1GB default.
        return 1_000_000_000

    if (
        scheduler == "distributed"
        or _env_get_int("POLARS_GPU_ENABLE_CUDA_MANAGED_MEMORY", default=1) == 0
    ):
        # Distributed execution requires a conservative
        # blocksize for now. We are also more conservative
        # when UVM is disabled.
        blocksize = int(device_size * 0.025)
    else:
        # Single-GPU execution can lean on UVM to
        # support a much larger blocksize.
        blocksize = int(device_size * 0.0625)

    # Use lower and upper bounds of 1GB and 10GB
    return min(max(blocksize, 1_000_000_000), 10_000_000_000)


@dataclasses.dataclass(frozen=True, eq=True)
class StreamingExecutor:
    """
    Configuration for the cudf-polars streaming executor.

    These options can be configured via environment variables
    with the prefix ``CUDF_POLARS__EXECUTOR__``.

    Parameters
    ----------
    scheduler
        The scheduler to use for the streaming executor. ``Scheduler.SYNCHRONOUS``
        by default.

        Note ``scheduler="distributed"`` requires a Dask cluster to be running.
    fallback_mode
        How to handle errors when the GPU engine fails to execute a query.
        ``StreamingFallbackMode.WARN`` by default.

        This can be set using the ``CUDF_POLARS__EXECUTOR__FALLBACK_MODE``
        environment variable.
    max_rows_per_partition
        The maximum number of rows to process per partition. 1_000_000 by default.
        When the number of rows exceeds this value, the query will be split into
        multiple partitions and executed in parallel.
    unique_fraction
        A dictionary mapping column names to floats between 0 and 1 (inclusive
        on the right).

        Each factor estimates the fractional number of unique values in the
        column. By default, ``1.0`` is used for any column not included in
        ``unique_fraction``.
    target_partition_size
        Target partition size, in bytes, for IO tasks. This configuration currently
        controls how large parquet files are split into multiple partitions.
        Files larger than ``target_partition_size`` bytes are split into multiple
        partitions.

        This can be set via

        - keyword argument to ``polars.GPUEngine``
        - the ``CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE`` environment variable

        By default, cudf-polars uses a target partition size that's a fraction
        of the device memory, where the fraction depends on the scheduler:

        - distributed: 1/40th of the device memory
        - synchronous: 1/16th of the device memory

        The optional pynvml dependency is used to query the device memory size. If
        pynvml is not available, a warning is emitted and the device size is assumed
        to be 12 GiB.

    groupby_n_ary
        The factor by which the number of partitions is decreased when performing
        a groupby on a partitioned column. For example, if a column has 64 partitions,
        it will first be reduced to ``ceil(64 / 32) = 2`` partitions.

        This is useful when the absolute number of partitions is large.
    broadcast_join_limit
        The maximum number of partitions to allow for the smaller table in
        a broadcast join.
    shuffle_method
        The method to use for shuffling data between workers. Defaults to
        'rapidsmpf' for distributed scheduler if available (otherwise 'tasks'),
        and 'tasks' for synchronous scheduler.
    rapidsmpf_spill
        Whether to wrap task arguments and output in objects that are
        spillable by 'rapidsmpf'.
    sink_to_directory
        Whether multi-partition sink operations should write to a directory
        rather than a single file. By default, this will be set to True for
        the 'distributed' scheduler and False otherwise. The 'distrubuted'
        scheduler does not currently support ``sink_to_directory=False``.

    Notes
    -----
    The streaming executor does not currently support profiling a query via
    the ``.profile()`` method. We recommend using nsys to profile queries
    with the 'synchronous' scheduler and Dask's built-in profiling tools
    with the 'distributed' scheduler.
    """

    _env_prefix = "CUDF_POLARS__EXECUTOR"

    name: Literal["streaming"] = dataclasses.field(default="streaming", init=False)
    scheduler: Scheduler = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__SCHEDULER",
            Scheduler.__call__,
            default=Scheduler.SYNCHRONOUS,
        )
    )
    fallback_mode: StreamingFallbackMode = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__FALLBACK_MODE",
            StreamingFallbackMode.__call__,
            default=StreamingFallbackMode.WARN,
        )
    )
    max_rows_per_partition: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__MAX_ROWS_PER_PARTITION", int, default=1_000_000
        )
    )
    unique_fraction: dict[str, float] = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__UNIQUE_FRACTION", json.loads, default={}
        )
    )
    target_partition_size: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__TARGET_PARTITION_SIZE", int, default=0
        )
    )
    groupby_n_ary: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__GROUPBY_N_ARY", int, default=32
        )
    )
    broadcast_join_limit: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__BROADCAST_JOIN_LIMIT", int, default=0
        )
    )
    shuffle_method: ShuffleMethod = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__SHUFFLE_METHOD",
            ShuffleMethod.__call__,
            default=ShuffleMethod.TASKS,
        )
    )
    rapidsmpf_spill: bool = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__RAPIDSMPF_SPILL", _bool_converter, default=False
        )
    )
    sink_to_directory: bool | None = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__SINK_TO_DIRECTORY", _bool_converter, default=None
        )
    )

    def __post_init__(self) -> None:  # noqa: D105
        # Handle shuffle_method defaults for streaming executor
        if self.shuffle_method is None:
            if self.scheduler == "distributed" and rapidsmpf_available():
                # For distributed scheduler, prefer rapidsmpf if available
                object.__setattr__(self, "shuffle_method", "rapidsmpf")
            else:
                object.__setattr__(self, "shuffle_method", "tasks")
        else:
            if (
                self.scheduler == "distributed"
                and self.shuffle_method == "rapidsmpf"
                and not rapidsmpf_available()
            ):
                raise ValueError(
                    "rapidsmpf shuffle method requested, but rapidsmpf is not installed"
                )
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
        object.__setattr__(self, "shuffle_method", ShuffleMethod(self.shuffle_method))

        if self.scheduler == "distributed":
            if self.sink_to_directory is False:
                raise ValueError(
                    "The distributed scheduler requires sink_to_directory=True"
                )
            object.__setattr__(self, "sink_to_directory", True)
        elif self.sink_to_directory is None:
            object.__setattr__(self, "sink_to_directory", False)

        # Type / value check everything else
        if not isinstance(self.max_rows_per_partition, int):
            raise TypeError("max_rows_per_partition must be an int")
        if not isinstance(self.unique_fraction, dict):
            raise TypeError("unique_fraction must be a dict of column name to float")
        if not isinstance(self.target_partition_size, int):
            raise TypeError("target_partition_size must be an int")
        if not isinstance(self.groupby_n_ary, int):
            raise TypeError("groupby_n_ary must be an int")
        if not isinstance(self.broadcast_join_limit, int):
            raise TypeError("broadcast_join_limit must be an int")
        if not isinstance(self.rapidsmpf_spill, bool):
            raise TypeError("rapidsmpf_spill must be bool")
        if not isinstance(self.sink_to_directory, bool):
            raise TypeError("sink_to_directory must be bool")

    def __hash__(self) -> int:  # noqa: D105
        # cardinality factory, a dict, isn't natively hashable. We'll dump it
        # to json and hash that.
        d = dataclasses.asdict(self)
        d["unique_fraction"] = json.dumps(d["unique_fraction"])
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
        :class:`~cudf_polars.utils.config.ParquetOptions` for more.
    executor
        The executor to use for the GPU engine. See :class:`~cudf_polars.utils.config.StreamingExecutor`
        and :class:`~cudf_polars.utils.config.InMemoryExecutor` for more.
    device
        The GPU used to run the query. If not provided, the
        query uses the current CUDA device.
    """

    raise_on_fail: bool = False
    parquet_options: ParquetOptions = dataclasses.field(default_factory=ParquetOptions)
    executor: StreamingExecutor | InMemoryExecutor = dataclasses.field(
        default_factory=StreamingExecutor
    )
    device: int | None = None

    @classmethod
    def from_polars_engine(
        cls, engine: polars.lazyframe.engine_config.GPUEngine
    ) -> Self:
        """Create a :class:`ConfigOptions` from a :class:`~polars.lazyframe.engine_config.GPUEngine`."""
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

        env_prefix = "CUDF_POLARS"
        user_executor = engine.config.get("executor")
        if user_executor is None:
            user_executor = os.environ.get(f"{env_prefix}__EXECUTOR", "streaming")
        user_executor_options = engine.config.get("executor_options", {})
        user_parquet_options = engine.config.get("parquet_options", {})
        # This is set in polars, and so can't be overridden by the environment
        user_raise_on_fail = engine.config.get("raise_on_fail", False)

        # Backward compatibility for "cardinality_factor"
        # TODO: Remove this in 25.10
        if "cardinality_factor" in user_executor_options:
            warnings.warn(
                "The 'cardinality_factor' configuration is deprecated. "
                "Please use 'unique_fraction' instead.",
                FutureWarning,
                stacklevel=2,
            )
            cardinality_factor = user_executor_options.pop("cardinality_factor")
            if "unique_fraction" not in user_executor_options:
                user_executor_options["unique_fraction"] = cardinality_factor

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
                user_executor_options = user_executor_options.copy()
                # Handle the interaction between the default shuffle method, the
                # scheduler, and whether rapidsmpf is available.
                env_shuffle_method = os.environ.get(
                    "CUDF_POLARS__EXECUTOR__SHUFFLE_METHOD", None
                )
                if env_shuffle_method is not None:
                    shuffle_method_default = ShuffleMethod(env_shuffle_method)
                else:
                    shuffle_method_default = None

                user_executor_options.setdefault(
                    "shuffle_method", shuffle_method_default
                )
                executor = StreamingExecutor(**user_executor_options)
            case _:  # pragma: no cover; Unreachable
                raise ValueError(f"Unsupported executor: {user_executor}")

        return cls(
            raise_on_fail=user_raise_on_fail,
            parquet_options=ParquetOptions(**user_parquet_options),
            executor=executor,
            device=engine.device,
        )
