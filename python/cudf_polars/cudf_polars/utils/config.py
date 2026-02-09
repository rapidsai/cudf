# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
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
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from rmm.pylibrmm.cuda_stream import CudaStreamFlags
from rmm.pylibrmm.cuda_stream_pool import CudaStreamPool

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    import polars.lazyframe.engine_config

    import rmm.mr


__all__ = [
    "Cluster",
    "ConfigOptions",
    "DynamicPlanningOptions",
    "InMemoryExecutor",
    "ParquetOptions",
    "Runtime",
    "Scheduler",  # Deprecated, kept for backward compatibility
    "ShuffleMethod",
    "ShufflerInsertionMethod",
    "StatsPlanningOptions",
    "StreamingExecutor",
    "StreamingFallbackMode",
]


def _env_get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (ValueError, TypeError):  # pragma: no cover
        return default  # pragma: no cover


@functools.cache
def get_device_handle() -> Any:
    # Gets called for each IR.do_evaluate node, so we'll cache it.
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
    except pynvml.NVMLError_NotSupported:  # pragma: no cover
        # System doesn't have proper "GPU memory".
        return None
    else:
        return handle


def get_total_device_memory() -> int | None:
    """Return the total memory of the current device."""
    import pynvml

    maybe_handle = get_device_handle()

    if maybe_handle is not None:
        try:
            return pynvml.nvmlDeviceGetMemoryInfo(maybe_handle).total
        except pynvml.NVMLError_NotSupported:  # pragma: no cover
            # System doesn't have proper "GPU memory".
            return None
    else:  # pragma: no cover
        return None


@functools.cache
def rapidsmpf_single_available() -> bool:  # pragma: no cover
    """Query whether rapidsmpf is available as a single-process shuffle method."""
    try:
        return importlib.util.find_spec("rapidsmpf.integrations.single") is not None
    except (ImportError, ValueError):
        return False


@functools.cache
def rapidsmpf_distributed_available() -> bool:  # pragma: no cover
    """Query whether rapidsmpf is available as a distributed shuffle method."""
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


class Runtime(str, enum.Enum):
    """
    The runtime to use for the streaming executor.

    * ``Runtime.TASKS`` : Use the task-based runtime.
      This is the default runtime.
    * ``Runtime.RAPIDSMPF`` : Use the coroutine-based streaming runtime (rapidsmpf).
      This runtime is experimental.
    """

    TASKS = "tasks"
    RAPIDSMPF = "rapidsmpf"


class Cluster(str, enum.Enum):
    """
    The cluster configuration for the streaming executor.

    * ``Cluster.SINGLE`` : Single-GPU execution. Currently uses a zero-dependency,
      synchronous, single-threaded task scheduler.
    * ``Cluster.DISTRIBUTED`` : Multi-GPU distributed execution. Currently
      uses a Dask-based distributed scheduler and requires an
      active Dask cluster.
    """

    SINGLE = "single"
    DISTRIBUTED = "distributed"


class Scheduler(str, enum.Enum):
    """
    **Deprecated**: Use :class:`Cluster` instead.

    The scheduler to use for the task-based streaming executor.

    * ``Scheduler.SYNCHRONOUS`` : Single-GPU execution (use ``Cluster.SINGLE`` instead)
    * ``Scheduler.DISTRIBUTED`` : Multi-GPU execution (use ``Cluster.DISTRIBUTED`` instead)
    """

    SYNCHRONOUS = "synchronous"
    DISTRIBUTED = "distributed"


class ShuffleMethod(str, enum.Enum):
    """
    The method to use for shuffling data between workers with the streaming executor.

    * ``ShuffleMethod.TASKS`` : Use the task-based shuffler.
    * ``ShuffleMethod.RAPIDSMPF`` : Use the rapidsmpf shuffler.
    * ``ShuffleMethod._RAPIDSMPF_SINGLE`` : Use the single-process rapidsmpf shuffler.

    With :class:`cudf_polars.utils.config.StreamingExecutor`, the default of ``None``
    will attempt to use ``ShuffleMethod.RAPIDSMPF`` for a distributed cluster,
    but will fall back to ``ShuffleMethod.TASKS`` if rapidsmpf is not installed.

    The user should **not** specify ``ShuffleMethod._RAPIDSMPF_SINGLE`` directly.
    A setting of ``ShuffleMethod.RAPIDSMPF`` will be converted to the single-process
    shuffler automatically when using single-GPU execution.
    """

    TASKS = "tasks"
    RAPIDSMPF = "rapidsmpf"
    _RAPIDSMPF_SINGLE = "rapidsmpf-single"


class ShufflerInsertionMethod(str, enum.Enum):
    """
    The method to use for inserting chunks into the rapidsmpf shuffler.

    * ``ShufflerInsertionMethod.INSERT_CHUNKS`` : Use insert_chunks for inserting data.
    * ``ShufflerInsertionMethod.CONCAT_INSERT`` : Use concat_insert for inserting data.

    Only applicable with the "rapidsmpf" shuffle method and the "tasks" runtime.
    """

    INSERT_CHUNKS = "insert_chunks"
    CONCAT_INSERT = "concat_insert"


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
    use_rapidsmpf_native
        Whether to use the native rapidsmpf node for parquet reading.
        This option is only used when the rapidsmpf runtime is enabled.
        Default is True.
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
    use_rapidsmpf_native: bool = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__USE_RAPIDSMPF_NATIVE",
            _bool_converter,
            default=True,
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
        if not isinstance(self.use_rapidsmpf_native, bool):
            raise TypeError("use_rapidsmpf_native must be a bool")


def default_blocksize(cluster: str) -> int:
    """Return the default blocksize."""
    device_size = get_total_device_memory()
    if device_size is None:  # pragma: no cover
        # System doesn't have proper "GPU memory".
        # Fall back to a conservative 1GB default.
        return 1_000_000_000

    if (
        cluster == "distributed"
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


@dataclasses.dataclass(frozen=True)
class StatsPlanningOptions:
    """
    Configuration for statistics-based query planning.

    These options can be configured via environment variables
    with the prefix ``CUDF_POLARS__EXECUTOR__STATS_PLANNING__``.

    Parameters
    ----------
    use_io_partitioning
        Whether to use estimated file-size statistics to calculate
        the ideal input-partition count for IO operations.
        This option currently applies to Parquet data only.
        Default is True.
    use_reduction_planning
        Whether to use estimated column statistics to calculate
        the output-partition count for reduction operations
        like `Distinct`, `GroupBy`, and `Select(unique)`.
        Default is False.
    use_join_heuristics
        Whether to use join heuristics to estimate row-count
        and unique-count statistics. Default is True.
        These statistics may only be collected when they are
        actually needed for query planning and when row-count
        statistics are available for the underlying datasource
        (e.g. Parquet and in-memory LazyFrame data).
    use_sampling
        Whether to sample real data to estimate unique-value
        statistics. Default is True.
        These statistics may only be collected when they are
        actually needed for query planning, and when the
        underlying datasource supports sampling (e.g. Parquet
        and in-memory LazyFrame data).
    default_selectivity
        The default selectivity of a predicate.
        Default is 0.8.
    """

    _env_prefix = "CUDF_POLARS__EXECUTOR__STATS_PLANNING"

    use_io_partitioning: bool = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__USE_IO_PARTITIONING", _bool_converter, default=True
        )
    )
    use_reduction_planning: bool = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__USE_REDUCTION_PLANNING", _bool_converter, default=False
        )
    )
    use_join_heuristics: bool = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__USE_JOIN_HEURISTICS", _bool_converter, default=True
        )
    )
    use_sampling: bool = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__USE_SAMPLING", _bool_converter, default=True
        )
    )
    default_selectivity: float = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__DEFAULT_SELECTIVITY", float, default=0.8
        )
    )

    def __post_init__(self) -> None:  # noqa: D105
        if not isinstance(self.use_io_partitioning, bool):
            raise TypeError("use_io_partitioning must be a bool")
        if not isinstance(self.use_reduction_planning, bool):
            raise TypeError("use_reduction_planning must be a bool")
        if not isinstance(self.use_join_heuristics, bool):
            raise TypeError("use_join_heuristics must be a bool")
        if not isinstance(self.use_sampling, bool):
            raise TypeError("use_sampling must be a bool")
        if not isinstance(self.default_selectivity, float):
            raise TypeError("default_selectivity must be a float")


@dataclasses.dataclass(frozen=True)
class DynamicPlanningOptions:
    """
    Configuration for dynamic shuffle planning.

    When enabled, shuffle decisions for GroupBy/Join/Unique operations
    are made at runtime by sampling real chunks.

    To enable dynamic planning, pass a ``DynamicPlanningOptions`` instance
    to ``StreamingExecutor(dynamic_planning=...)``. To disable it, pass
    ``None`` (the default).

    These options can be configured via environment variables
    with the prefix ``CUDF_POLARS__EXECUTOR__DYNAMIC_PLANNING__``.

    .. note::
        Dynamic planning is not yet implemented. These options are
        reserved for future use and currently have no effect.

    Parameters
    ----------
    sample_chunk_count
        The maximum number of chunks to sample before deciding whether
        to shuffle. A higher value provides more accurate estimates but
        increases latency before the shuffle decision is made.
        Default is 2.
    """

    _env_prefix = "CUDF_POLARS__EXECUTOR__DYNAMIC_PLANNING"

    sample_chunk_count: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__SAMPLE_CHUNK_COUNT", int, default=2
        )
    )

    def __post_init__(self) -> None:  # noqa: D105
        if not isinstance(self.sample_chunk_count, int):
            raise TypeError("sample_chunk_count must be an int")
        if self.sample_chunk_count < 1:
            raise ValueError("sample_chunk_count must be at least 1")


@dataclasses.dataclass(frozen=True, eq=True)
class MemoryResourceConfig:
    """
    Configuration for the default memory resource.

    Parameters
    ----------
    qualname
        The fully qualified name of the memory resource class to use.
    options
        This can be either a dictionary representing the options to pass
        to the memory resource class, or, a dictionary representing a
        nested memory resource configuration. The presence of "qualname"
        field indicates a nested memory resource configuration.

    Examples
    --------
    Create a memory resource config for a single memory resource:
    >>> MemoryResourceConfig(
    ...     qualname="rmm.mr.CudaAsyncMemoryResource",
    ...     options={"initial_pool_size": 100},
    ... )

    Create a memory resource config for a nested memory resource configuration:
    >>> MemoryResourceConfig(
    ...     qualname="rmm.mr.PrefetchResourceAdaptor",
    ...     options={
    ...         "upstream_mr": {
    ...             "qualname": "rmm.mr.PoolMemoryResource",
    ...             "options": {
    ...                 "upstream_mr": {
    ...                     "qualname": "rmm.mr.ManagedMemoryResource",
    ...                 },
    ...                 "initial_pool_size": 256,
    ...             },
    ...         }
    ...     },
    ... )
    """

    _env_prefix = "CUDF_POLARS__MEMORY_RESOURCE_CONFIG"
    qualname: str = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__QUALNAME",
            str,
            # We shouldn't reach here if qualname isn't set in the environment.
            default=None,  # type: ignore[assignment]
        )
    )
    options: dict[str, Any] | None = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__OPTIONS",
            json.loads,
            default=None,
        )
    )

    def __post_init__(self) -> None:
        if self.qualname.count(".") < 1:
            raise ValueError(
                f"MemoryResourceConfig.qualname '{self.qualname}' must be a fully qualified name to a class, including the module name."
            )

    def create_memory_resource(self) -> rmm.mr.DeviceMemoryResource:
        """Create a memory resource from the configuration."""

        def create_mr(
            qualname: str, options: dict[str, Any] | None
        ) -> rmm.mr.DeviceMemoryResource:
            module_name, class_name = qualname.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls(**options or {})

        def process_options(opts: dict[str, Any] | None) -> dict[str, Any]:
            if opts is None:
                return {}

            processed = {}
            for key, value in opts.items():
                if isinstance(value, dict) and "qualname" in value:
                    # This is a nested memory resource config
                    nested_qualname = value["qualname"]
                    nested_options = process_options(value.get("options"))
                    processed[key] = create_mr(nested_qualname, nested_options)
                else:
                    processed[key] = value
            return processed

        # Create the top-level memory resource
        return create_mr(self.qualname, process_options(self.options))

    def __hash__(self) -> int:
        return hash((self.qualname, json.dumps(self.options, sort_keys=True)))


@dataclasses.dataclass(frozen=True, eq=True)
class StreamingExecutor:
    """
    Configuration for the cudf-polars streaming executor.

    These options can be configured via environment variables
    with the prefix ``CUDF_POLARS__EXECUTOR__``.

    Parameters
    ----------
    runtime
        The runtime to use for the streaming executor.
        ``Runtime.TASKS`` by default.
    cluster
        The cluster configuration for the streaming executor.
        ``Cluster.SINGLE`` by default.

        This setting applies to both task-based and rapidsmpf execution modes:

        * ``Cluster.SINGLE``: Single-GPU execution
        * ``Cluster.DISTRIBUTED``: Multi-GPU distributed execution (requires
          an active Dask cluster)

    scheduler
        **Deprecated**: Use ``cluster`` instead.

        For backward compatibility:
        * ``Scheduler.SYNCHRONOUS`` maps to ``Cluster.SINGLE``
        * ``Scheduler.DISTRIBUTED`` maps to ``Cluster.DISTRIBUTED``
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
        of the device memory, where the fraction depends on the cluster:

        - distributed: 1/40th of the device memory
        - single: 1/16th of the device memory

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
        'rapidsmpf' for distributed cluster if available (otherwise 'tasks'),
        and 'tasks' for single-GPU cluster.
    shuffler_insertion_method
        The method to use for inserting chunks with the rapidsmpf shuffler.
        Can be 'insert_chunks' (default) or 'concat_insert'.

        Only applicable with ``shuffle_method="rapidsmpf"`` and ``runtime="tasks"``.
    rapidsmpf_spill
        Whether to wrap task arguments and output in objects that are
        spillable by 'rapidsmpf'.
    client_device_threshold
        Threshold for spilling data from device memory in rapidsmpf.
        Default is 50% of device memory on the client process.
        This argument is only used by the "rapidsmpf" runtime.
    sink_to_directory
        Whether multi-partition sink operations should write to a directory
        rather than a single file. By default, this will be set to True for
        the 'distributed' cluster and False otherwise. The 'distributed'
        cluster does not currently support ``sink_to_directory=False``.
    stats_planning
        Options controlling statistics-based query planning. See
        :class:`~cudf_polars.utils.config.StatsPlanningOptions` for more.
    dynamic_planning
        Options controlling dynamic shuffle planning. See
        :class:`~cudf_polars.utils.config.DynamicPlanningOptions` for more.

        .. note::
            Dynamic planning is not yet implemented. These options are
            reserved for future use and currently have no effect.
    max_io_threads
        Maximum number of IO threads for the rapidsmpf runtime. Default is 2.
        This controls the parallelism of IO operations when reading data.
    spill_to_pinned_memory
        Whether RapidsMPF should spill to pinned host memory when available,
        or use regular pageable host memory. Pinned host memory offers higher
        bandwidth and lower latency for device to host transfers compared to
        regular pageable host memory.

    Notes
    -----
    The streaming executor does not currently support profiling a query via
    the ``.profile()`` method. We recommend using nsys to profile queries
    with single-GPU execution and Dask's built-in profiling tools
    with distributed execution.
    """

    _env_prefix = "CUDF_POLARS__EXECUTOR"

    name: Literal["streaming"] = dataclasses.field(default="streaming", init=False)
    runtime: Runtime = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__RUNTIME",
            Runtime.__call__,
            default=Runtime.TASKS,
        )
    )
    cluster: Cluster | None = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__CLUSTER",
            Cluster.__call__,
            default=None,
        )
    )
    scheduler: Scheduler | None = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__SCHEDULER",
            Scheduler.__call__,
            default=None,
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
    shuffler_insertion_method: ShufflerInsertionMethod = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__SHUFFLER_INSERTION_METHOD",
            ShufflerInsertionMethod.__call__,
            default=ShufflerInsertionMethod.INSERT_CHUNKS,
        )
    )
    rapidsmpf_spill: bool = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__RAPIDSMPF_SPILL", _bool_converter, default=False
        )
    )
    client_device_threshold: float = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__CLIENT_DEVICE_THRESHOLD", float, default=0.5
        )
    )
    sink_to_directory: bool | None = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__SINK_TO_DIRECTORY", _bool_converter, default=None
        )
    )
    stats_planning: StatsPlanningOptions = dataclasses.field(
        default_factory=StatsPlanningOptions
    )
    dynamic_planning: DynamicPlanningOptions | None = None
    max_io_threads: int = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__MAX_IO_THREADS", int, default=2
        )
    )
    spill_to_pinned_memory: bool = dataclasses.field(
        default_factory=_make_default_factory(
            f"{_env_prefix}__SPILL_TO_PINNED_MEMORY", bool, default=False
        )
    )

    def __post_init__(self) -> None:  # noqa: D105
        # Check for rapidsmpf runtime
        if self.runtime == "rapidsmpf":  # pragma: no cover; requires rapidsmpf runtime
            if not rapidsmpf_single_available():
                raise ValueError("The rapidsmpf streaming engine requires rapidsmpf.")
            if self.shuffle_method == "tasks":
                raise ValueError(
                    "The rapidsmpf streaming engine does not support task-based shuffling."
                )
            object.__setattr__(self, "shuffle_method", "rapidsmpf")

        # Handle backward compatibility for deprecated scheduler parameter
        if self.scheduler is not None:
            if self.cluster is not None:
                raise ValueError(
                    "Cannot specify both 'scheduler' and 'cluster'. "
                    "The 'scheduler' parameter is deprecated. "
                    "Please use only 'cluster' instead."
                )
            else:
                warnings.warn(
                    """The 'scheduler' parameter is deprecated. Please use 'cluster' instead.
                    Use 'cluster="single"' instead of 'scheduler="synchronous"' and "
                    'cluster="distributed"' instead of 'scheduler="distributed"'.""",
                    FutureWarning,
                    stacklevel=2,
                )
            # Map old scheduler values to new cluster values
            if self.scheduler == "synchronous":
                object.__setattr__(self, "cluster", Cluster.SINGLE)
            elif self.scheduler == "distributed":
                object.__setattr__(self, "cluster", Cluster.DISTRIBUTED)
            # Clear scheduler to avoid confusion
            object.__setattr__(self, "scheduler", None)
        elif self.cluster is None:
            object.__setattr__(self, "cluster", Cluster.SINGLE)
        assert self.cluster is not None, "Expected cluster to be set."

        # Warn loudly that multi-GPU execution is under construction
        # for the rapidsmpf runtime
        if self.cluster == "distributed" and self.runtime == "rapidsmpf":
            warnings.warn(
                "UNDER CONSTRUCTION!!!"
                "The rapidsmpf runtime does NOT support distributed execution yet. "
                "Use at your own risk!!!",
                stacklevel=2,
            )

        # Handle shuffle_method defaults for streaming executor
        if self.shuffle_method is None:
            if self.cluster == "distributed" and rapidsmpf_distributed_available():
                # For distributed cluster, prefer rapidsmpf if available
                object.__setattr__(self, "shuffle_method", "rapidsmpf")
            else:
                # Otherwise, use task-based shuffle for now.
                # TODO: Evaluate single-process shuffle by default.
                object.__setattr__(self, "shuffle_method", "tasks")
        elif self.shuffle_method == "rapidsmpf-single":
            # The user should NOT specify "rapidsmpf-single" directly.
            raise ValueError("rapidsmpf-single is not a supported shuffle method.")
        elif self.shuffle_method == "rapidsmpf":
            # Check that we have rapidsmpf installed
            if self.cluster == "distributed" and not rapidsmpf_distributed_available():
                raise ValueError(
                    "rapidsmpf shuffle method requested, but rapidsmpf.integrations.dask is not installed."
                )
            elif self.cluster == "single" and not rapidsmpf_single_available():
                raise ValueError(
                    "rapidsmpf shuffle method requested, but rapidsmpf is not installed."
                )
            # Select "rapidsmpf-single" for single-GPU
            if self.cluster == "single":
                object.__setattr__(self, "shuffle_method", "rapidsmpf-single")

        # frozen dataclass, so use object.__setattr__
        object.__setattr__(
            self, "fallback_mode", StreamingFallbackMode(self.fallback_mode)
        )
        if self.target_partition_size == 0:
            object.__setattr__(
                self,
                "target_partition_size",
                default_blocksize(self.cluster),
            )
        if self.broadcast_join_limit == 0:
            object.__setattr__(
                self,
                "broadcast_join_limit",
                # Usually better to avoid shuffling for single gpu with UVM
                2 if self.cluster == "distributed" else 32,
            )
        object.__setattr__(self, "cluster", Cluster(self.cluster))
        object.__setattr__(self, "shuffle_method", ShuffleMethod(self.shuffle_method))
        object.__setattr__(
            self,
            "shuffler_insertion_method",
            ShufflerInsertionMethod(self.shuffler_insertion_method),
        )

        # Make sure stats_planning is a dataclass
        if isinstance(self.stats_planning, dict):
            object.__setattr__(
                self,
                "stats_planning",
                StatsPlanningOptions(**self.stats_planning),
            )

        # Handle dynamic_planning.
        # Can be None, dict, or DynamicPlanningOptions
        if isinstance(self.dynamic_planning, dict):
            object.__setattr__(
                self,
                "dynamic_planning",
                DynamicPlanningOptions(**self.dynamic_planning),
            )

        if self.cluster == "distributed":
            if self.sink_to_directory is False:
                raise ValueError(
                    "The distributed cluster requires sink_to_directory=True"
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
        if not isinstance(self.client_device_threshold, float):
            raise TypeError("client_device_threshold must be a float")
        if not isinstance(self.max_io_threads, int):
            raise TypeError("max_io_threads must be an int")
        if not isinstance(self.spill_to_pinned_memory, bool):
            raise TypeError("spill_to_pinned_memory must be bool")

        # RapidsMPF spill is only supported for distributed clusters for now.
        # This is because the spilling API is still within the RMPF-Dask integration.
        # (See https://github.com/rapidsai/rapidsmpf/issues/439)
        if self.cluster == "single" and self.rapidsmpf_spill:  # pragma: no cover
            raise ValueError(
                "rapidsmpf_spill is not supported for single-GPU execution."
            )

    def __hash__(self) -> int:  # noqa: D105
        # cardinality factory, a dict, isn't natively hashable. We'll dump it
        # to json and hash that.
        d = dataclasses.asdict(self)
        d["unique_fraction"] = json.dumps(d["unique_fraction"])
        d["stats_planning"] = json.dumps(d["stats_planning"])
        d["dynamic_planning"] = json.dumps(d["dynamic_planning"])
        return hash(tuple(sorted(d.items())))


@dataclasses.dataclass(frozen=True, eq=True)
class InMemoryExecutor:
    """
    Configuration for the cudf-polars in-memory executor.

    The in-memory executor only supports single-GPU execution.
    """

    name: Literal["in-memory"] = dataclasses.field(default="in-memory", init=False)


@dataclasses.dataclass(frozen=True, eq=True)
class CUDAStreamPoolConfig:
    """
    Configuration for the CUDA stream pool.

    Parameters
    ----------
    pool_size
        The size of the CUDA stream pool.
    flags
        The flags to use for the CUDA stream pool.
    """

    pool_size: int = 16
    flags: CudaStreamFlags = CudaStreamFlags.NON_BLOCKING

    def build(self) -> CudaStreamPool:
        return CudaStreamPool(
            pool_size=self.pool_size,
            flags=self.flags,
        )


class CUDAStreamPolicy(str, enum.Enum):
    """
    The policy to use for acquiring new CUDA streams.

    * ``CUDAStreamPolicy.DEFAULT`` : Use the default CUDA stream.
    * ``CUDAStreamPolicy.NEW`` : Create a new CUDA stream.
    """

    DEFAULT = "default"
    NEW = "new"


def _convert_cuda_stream_policy(
    user_cuda_stream_policy: dict | str,
) -> CUDAStreamPolicy | CUDAStreamPoolConfig:
    match user_cuda_stream_policy:
        case "default" | "new":
            return CUDAStreamPolicy(user_cuda_stream_policy)
        case "pool":
            return CUDAStreamPoolConfig()
        case dict():
            return CUDAStreamPoolConfig(**user_cuda_stream_policy)
        case str():
            # assume it's a JSON encoded CUDAStreamPoolConfig
            try:
                d = json.loads(user_cuda_stream_policy)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Invalid CUDA stream policy: '{user_cuda_stream_policy}'"
                ) from None
            match d:
                case {"pool_size": int(), "flags": int()}:
                    return CUDAStreamPoolConfig(
                        pool_size=d["pool_size"], flags=CudaStreamFlags(d["flags"])
                    )
                case {"pool_size": int(), "flags": str()}:
                    # convert the string names to enums
                    return CUDAStreamPoolConfig(
                        pool_size=d["pool_size"],
                        flags=CudaStreamFlags(CudaStreamFlags.__members__[d["flags"]]),
                    )
                case _:
                    try:
                        return CUDAStreamPoolConfig(**d)
                    except TypeError:
                        raise ValueError(
                            f"Invalid CUDA stream policy: {user_cuda_stream_policy}"
                        ) from None


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
    cuda_stream_policy
        The policy to use for acquiring new CUDA streams. See :class:`~cudf_polars.utils.config.CUDAStreamPolicy` for more.
    """

    raise_on_fail: bool = False
    parquet_options: ParquetOptions = dataclasses.field(default_factory=ParquetOptions)
    executor: StreamingExecutor | InMemoryExecutor = dataclasses.field(
        default_factory=StreamingExecutor
    )
    device: int | None = None
    memory_resource_config: MemoryResourceConfig | None = None
    cuda_stream_policy: CUDAStreamPolicy | CUDAStreamPoolConfig = dataclasses.field(
        default_factory=_make_default_factory(
            "CUDF_POLARS__CUDA_STREAM_POLICY",
            CUDAStreamPolicy.__call__,
            default=CUDAStreamPolicy.DEFAULT,
        )
    )

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
            "memory_resource_config",
            "cuda_stream_policy",
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
        if user_parquet_options is None:
            user_parquet_options = {}
        # This is set in polars, and so can't be overridden by the environment
        user_raise_on_fail = engine.config.get("raise_on_fail", False)
        user_memory_resource_config = engine.config.get("memory_resource_config", None)
        if user_memory_resource_config is None and (
            os.environ.get(f"{MemoryResourceConfig._env_prefix}__QUALNAME", "") != ""
        ):
            # We'll pick up the qualname / options from the environment.
            user_memory_resource_config = MemoryResourceConfig()
        elif isinstance(user_memory_resource_config, dict):
            user_memory_resource_config = MemoryResourceConfig(
                **user_memory_resource_config
            )

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
                # cluster, and whether rapidsmpf is available.
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

                # Handle dynamic_planning: check user config, then env var
                user_dynamic_planning = user_executor_options.get(
                    "dynamic_planning", None
                )
                if user_dynamic_planning is None:
                    env_dynamic_planning = os.environ.get(
                        "CUDF_POLARS__EXECUTOR__DYNAMIC_PLANNING", "0"
                    )
                    if _bool_converter(env_dynamic_planning):
                        user_executor_options["dynamic_planning"] = (
                            DynamicPlanningOptions()
                        )

                executor = StreamingExecutor(**user_executor_options)
            case _:  # pragma: no cover; Unreachable
                raise ValueError(f"Unsupported executor: {user_executor}")

        kwargs = {
            "raise_on_fail": user_raise_on_fail,
            "parquet_options": ParquetOptions(**user_parquet_options),
            "executor": executor,
            "device": engine.device,
            "memory_resource_config": user_memory_resource_config,
        }

        # Handle "cuda-stream-policy".
        # The default will depend on the runtime and executor.
        user_cuda_stream_policy = engine.config.get(
            "cuda_stream_policy", None
        ) or os.environ.get("CUDF_POLARS__CUDA_STREAM_POLICY", None)

        cuda_stream_policy: CUDAStreamPolicy | CUDAStreamPoolConfig

        if user_cuda_stream_policy is None:
            if (
                executor.name == "streaming" and executor.runtime == Runtime.RAPIDSMPF
            ):  # pragma: no cover; requires rapidsmpf runtime
                # the rapidsmpf runtime defaults to using a stream pool
                cuda_stream_policy = CUDAStreamPoolConfig()
            else:
                # everything else defaults to the default stream
                cuda_stream_policy = CUDAStreamPolicy.DEFAULT
        else:
            cuda_stream_policy = _convert_cuda_stream_policy(user_cuda_stream_policy)

        # Pool policy is only supported by the rapidsmpf runtime.
        if isinstance(cuda_stream_policy, CUDAStreamPoolConfig) and (
            (executor.name != "streaming")
            or (executor.name == "streaming" and executor.runtime != Runtime.RAPIDSMPF)
        ):
            raise ValueError(
                "CUDAStreamPolicy.POOL is only supported by the rapidsmpf runtime."
            )

        kwargs["cuda_stream_policy"] = cuda_stream_policy

        return cls(**kwargs)
