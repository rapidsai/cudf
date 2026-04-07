# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: RUF009  -- _opt() returns dataclasses.field(), not a mutable default
"""Unified streaming options for the RapidsMPF frontend."""

from __future__ import annotations

import argparse
import dataclasses
import json
import textwrap
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from rapidsmpf.config import Options, get_environment_variables

if TYPE_CHECKING:
    from cudf_polars.utils.config import (
        DynamicPlanningOptions,
        MemoryResourceConfig,
        ParquetOptions,
    )

__all__: list[str] = [
    "UNSPECIFIED",
    "StreamingOptions",
]


class _Unspecified:
    """Sentinel value meaning "fall back to env var, then built-in default"."""

    _instance: _Unspecified | None = None

    def __new__(cls) -> _Unspecified:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSPECIFIED"


UNSPECIFIED = _Unspecified()
"""Sentinel default for all option fields.

When a field is left at ``UNSPECIFIED``, its value is resolved in order:
  1. The corresponding environment variable (if set).
  2. The built-in default of the underlying library (rapidsmpf C++ or cudf-polars).
"""


def _opt(category: str) -> Any:
    """
    Factory for ``StreamingOptions`` fields with category metadata.

    Parameters
    ----------
    category
        Which ``_to_*`` method claims this field: ``"rapidsmpf"``,
        ``"executor"``, or ``"engine"``.
    """
    return dataclasses.field(default=UNSPECIFIED, metadata={"category": category})


@dataclasses.dataclass
class StreamingOptions:
    """
    High-level configuration for the cudf-polars streaming executor and RapidsMPF.

    Options are grouped into three categories:
      - **RapidsMPF**: runtime and memory behavior (e.g. spilling, threading).
      - **Executor**: query execution and partitioning behavior.
      - **Engine**: Polars integration and IO configuration.

    All fields default to :data:`UNSPECIFIED` and follow this precedence:
      1. Explicit value.
      2. Environment variable.
      3. Built-in default.

    Parameters
    ----------
    num_streaming_threads
        Threads used to execute coroutines.
        Env: ``RAPIDSMPF_NUM_STREAMING_THREADS``.
        Default: ``1``.
        Category: rapidsmpf.
    num_streams
        CUDA streams for concurrent GPU execution.
        Env: ``RAPIDSMPF_NUM_STREAMS``.
        Default: ``16``.
        Category: rapidsmpf.
    log
        Log level (``"NONE"``, ``"PRINT"``, ``"WARN"``, ``"INFO"``,
        ``"DEBUG"``, ``"TRACE"``).
        Env: ``RAPIDSMPF_LOG``.
        Default: ``"WARN"``.
        Category: rapidsmpf.
    statistics
        Enable performance metrics.
        Env: ``RAPIDSMPF_STATISTICS``.
        Default: ``False``.
        Category: rapidsmpf.
    memory_reserve_timeout
        Timeout for memory reservations (e.g. ``"100ms"``).
        Env: ``RAPIDSMPF_MEMORY_RESERVE_TIMEOUT``.
        Default: ``"100ms"``.
        Category: rapidsmpf.
    allow_overbooking_by_default
        Allow overallocation in reservation APIs.
        Env: ``RAPIDSMPF_ALLOW_OVERBOOKING_BY_DEFAULT``.
        Default: ``True``.
        Category: rapidsmpf.
    pinned_memory
        Enable pinned host memory.
        Env: ``RAPIDSMPF_PINNED_MEMORY``.
        Default: ``False``.
        Category: rapidsmpf.
    pinned_initial_pool_size
        Initial pinned memory pool size (bytes).
        Env: ``RAPIDSMPF_PINNED_INITIAL_POOL_SIZE``.
        Default: ``0``.
        Category: rapidsmpf.
    spill_device_limit
        Device memory soft limit before spilling (e.g. ``"80%"`` or bytes).
        Env: ``RAPIDSMPF_SPILL_DEVICE_LIMIT``.
        Default: ``"80%"``.
        Category: rapidsmpf.
    periodic_spill_check
        Interval between spill checks (e.g. ``"1ms"``).
        Env: ``RAPIDSMPF_PERIODIC_SPILL_CHECK``.
        Default: ``"1ms"``.
        Category: rapidsmpf.
    num_py_executors
        Workers for the internal Python ``ThreadPoolExecutor``.
        Env: ``CUDF_POLARS__EXECUTOR__NUM_PY_EXECUTORS``.
        Default: ``1``.
        Category: executor.
    fallback_mode
        Fallback behavior (``"warn"``, ``"raise"``, ``"silent"``).
        Env: ``CUDF_POLARS__EXECUTOR__FALLBACK_MODE``.
        Default: ``"warn"``.
        Category: executor.
    max_rows_per_partition
        Maximum rows per partition.
        Env: ``CUDF_POLARS__EXECUTOR__MAX_ROWS_PER_PARTITION``.
        Default: ``1_000_000``.
        Category: executor.
    broadcast_join_limit
        Max partitions for broadcast joins.
        Env: ``CUDF_POLARS__EXECUTOR__BROADCAST_JOIN_LIMIT``.
        Default: auto.
        Category: executor.
    target_partition_size
        Target IO partition size (bytes). ``0`` = auto.
        Env: ``CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE``.
        Default: auto.
        Category: executor.
    dynamic_planning
        Dynamic planning config, dict or
        :class:`~cudf_polars.utils.config.DynamicPlanningOptions`. ``None`` disables.
        Env: ``CUDF_POLARS__EXECUTOR__DYNAMIC_PLANNING``.
        Default: enabled.
        Category: executor.
    unique_fraction
        Per-column uniqueness estimate (0-1). Defaults to ``1.0``.
        Env: ``CUDF_POLARS__EXECUTOR__UNIQUE_FRACTION``.
        Default: ``{}``.
        Category: executor.
    raise_on_fail
        Raise instead of falling back to CPU.
        Default: ``False``.
        Category: engine.
    parquet_options
        Parquet configuration, dict or
        :class:`~cudf_polars.utils.config.ParquetOptions`.
        Env: ``CUDF_POLARS__PARQUET_OPTIONS__*``.
        Category: engine.
    memory_resource_config
        RMM configuration, dict or
        :class:`~cudf_polars.utils.config.MemoryResourceConfig`.
        Env: ``CUDF_POLARS__MEMORY_RESOURCE_CONFIG__*``.
        Category: engine.
    cuda_stream_policy
        CUDA stream policy (``"default"``, ``"new"``, ``"pool"`` or config dict).
        Env: ``CUDF_POLARS__CUDA_STREAM_POLICY``.
        Category: engine.

    Examples
    --------
    >>> StreamingOptions(num_streaming_threads=8, log="DEBUG", fallback_mode="silent")
    StreamingOptions(...)
    """

    # ---- RapidsMPF runtime ----
    num_streaming_threads: int | _Unspecified = _opt("rapidsmpf")
    num_streams: int | _Unspecified = _opt("rapidsmpf")
    log: Literal["NONE", "PRINT", "WARN", "INFO", "DEBUG", "TRACE"] | _Unspecified = (
        _opt("rapidsmpf")
    )
    statistics: bool | _Unspecified = _opt("rapidsmpf")
    memory_reserve_timeout: str | _Unspecified = _opt("rapidsmpf")
    allow_overbooking_by_default: bool | _Unspecified = _opt("rapidsmpf")
    pinned_memory: bool | _Unspecified = _opt("rapidsmpf")
    pinned_initial_pool_size: int | _Unspecified = _opt("rapidsmpf")
    spill_device_limit: str | _Unspecified = _opt("rapidsmpf")
    periodic_spill_check: str | _Unspecified = _opt("rapidsmpf")
    # ---- Executor ----
    num_py_executors: int | _Unspecified = _opt("executor")
    fallback_mode: str | _Unspecified = _opt("executor")
    max_rows_per_partition: int | _Unspecified = _opt("executor")
    broadcast_join_limit: int | _Unspecified = _opt("executor")
    target_partition_size: int | _Unspecified = _opt("executor")
    dynamic_planning: dict[str, Any] | DynamicPlanningOptions | None | _Unspecified = (
        _opt("executor")
    )
    unique_fraction: dict[str, float] | _Unspecified = _opt("executor")
    # ---- Engine ----
    raise_on_fail: bool | _Unspecified = _opt("engine")
    parquet_options: dict[str, Any] | ParquetOptions | _Unspecified = _opt("engine")
    memory_resource_config: dict[str, Any] | MemoryResourceConfig | _Unspecified = _opt(
        "engine"
    )
    cuda_stream_policy: (
        Literal["default", "new", "pool"] | dict[str, Any] | _Unspecified
    ) = _opt("engine")

    _VALID_LOG_LEVELS: ClassVar[frozenset[str]] = frozenset(
        {"NONE", "PRINT", "WARN", "INFO", "DEBUG", "TRACE"}
    )
    _VALID_FALLBACK_MODES: ClassVar[frozenset[str]] = frozenset(
        {"warn", "raise", "silent"}
    )

    def __post_init__(self) -> None:
        """Validate field values after construction."""
        if (
            not isinstance(self.num_streaming_threads, _Unspecified)
            and self.num_streaming_threads <= 0
        ):
            raise ValueError(
                f"num_streaming_threads must be > 0, got {self.num_streaming_threads}"
            )
        if not isinstance(self.num_streams, _Unspecified) and self.num_streams <= 0:
            raise ValueError(f"num_streams must be > 0, got {self.num_streams}")
        if (
            not isinstance(self.log, _Unspecified)
            and self.log not in self._VALID_LOG_LEVELS
        ):
            raise ValueError(
                f"log must be one of {sorted(self._VALID_LOG_LEVELS)}, got {self.log!r}"
            )
        if (
            not isinstance(self.fallback_mode, _Unspecified)
            and self.fallback_mode not in self._VALID_FALLBACK_MODES
        ):
            raise ValueError(
                f"fallback_mode must be one of {sorted(self._VALID_FALLBACK_MODES)}, "
                f"got {self.fallback_mode!r}"
            )

    # ------------------------------------------------------------------
    # Conversion helpers used by the engines
    # ------------------------------------------------------------------

    def to_rapidsmpf_options(self) -> Options:
        """
        Build a ``rapidsmpf.config.Options`` from the RapidsMPF fields.

        Fields that are :data:`UNSPECIFIED` fall back to the corresponding
        ``RAPIDSMPF_*`` environment variable (if set); otherwise the rapidsmpf
        C++ library applies its own built-in default.
        """
        # get_environment_variables() returns uppercase keys,
        # e.g. {"NUM_STREAMING_THREADS": "4"}.
        env = get_environment_variables()

        opts: dict[str, str] = {}
        for f in dataclasses.fields(self):
            if f.metadata.get("category") != "rapidsmpf":
                continue
            v = getattr(self, f.name)
            if isinstance(v, _Unspecified):
                env_key = f.name.upper()
                if env_key in env:
                    opts[f.name] = env[env_key]
                # else: omit entirely → rapidsmpf C++ uses its built-in default
            else:
                opts[f.name] = str(v)

        return Options(opts)

    def to_executor_options(self) -> dict[str, Any]:
        """
        Build a ``StreamingExecutor`` kwargs dict from the executor fields.

        Only fields that are not :data:`UNSPECIFIED` are included.
        ``StreamingExecutor`` reads ``CUDF_POLARS__EXECUTOR__*`` environment
        variables for any omitted fields.
        """
        opts: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.metadata.get("category") != "executor":
                continue
            v = getattr(self, f.name)
            if not isinstance(v, _Unspecified):
                opts[f.name] = v
        return opts

    def to_engine_options(self) -> dict[str, Any]:
        """
        Build a ``pl.GPUEngine`` kwargs dict from the engine fields.

        Only fields that are not :data:`UNSPECIFIED` are included.
        ``ConfigOptions.from_polars_engine`` handles environment variables for
        any omitted fields.
        """
        opts: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.metadata.get("category") != "engine":
                continue
            v = getattr(self, f.name)
            if not isinstance(v, _Unspecified):
                opts[f.name] = v
        return opts

    def to_dict(self) -> dict[str, Any]:
        """
        Return all explicitly-set fields as a plain dictionary.

        Fields that are :data:`UNSPECIFIED` are omitted. The result can be
        round-tripped via :meth:`from_dict`.

        Returns
        -------
        dict
            Mapping of field name to value for every non-UNSPECIFIED field.

        Examples
        --------
        >>> StreamingOptions(fallback_mode="silent").to_dict()
        {'fallback_mode': 'silent'}
        >>> StreamingOptions.from_dict(
        ...     StreamingOptions(fallback_mode="silent").to_dict()
        ... )  # doctest: +ELLIPSIS
        StreamingOptions(...)
        """
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if not isinstance(getattr(self, f.name), _Unspecified)
        }

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StreamingOptions:
        """
        Build a :class:`StreamingOptions` from a plain dictionary.

        Keys must be field names of :class:`StreamingOptions`. Values of ``None``
        and missing keys both leave the corresponding field at :data:`UNSPECIFIED`.

        Parameters
        ----------
        d
            Flat dictionary of option name to value. Unknown keys raise
            :exc:`TypeError`.

        Returns
        -------
        A new :class:`StreamingOptions` instance.

        Raises
        ------
        TypeError
            If ``d`` contains a key that is not a known field.

        Examples
        --------
        >>> StreamingOptions.from_dict(
        ...     {"fallback_mode": "silent", "num_streaming_threads": 4}
        ... )  # doctest: +ELLIPSIS
        StreamingOptions(...)
        >>> StreamingOptions.from_dict({})  # all fields UNSPECIFIED
        StreamingOptions(...)
        """
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = d.keys() - known
        if unknown:
            raise TypeError(
                f"StreamingOptions.from_dict() got unknown field(s): "
                f"{', '.join(sorted(unknown))}"
            )
        kwargs = {k: (UNSPECIFIED if v is None else v) for k, v in d.items()}
        return cls(**kwargs)

    @classmethod
    def from_argparse(cls, args: argparse.Namespace) -> StreamingOptions:
        """
        Build a :class:`StreamingOptions` from a parsed :class:`argparse.Namespace`.

        Designed to work with namespaces produced by :func:`argparse.ArgumentParser`
        parsers that have been augmented with :meth:`add_cli_args`. Fields not present
        in the namespace (or set to ``None``) remain :data:`UNSPECIFIED`.

        Parameters
        ----------
        args
            Parsed namespace, typically from ``parser.parse_args()``.

        Returns
        -------
        A new :class:`StreamingOptions` instance.

        Examples
        --------
        >>> parser = argparse.ArgumentParser()
        >>> StreamingOptions.add_cli_args(parser)
        >>> opts = StreamingOptions.from_argparse(parser.parse_args([]))
        """

        def _get(attr: str) -> Any:
            """Return the attr value, or UNSPECIFIED if absent or None."""
            v = getattr(args, attr, None)
            return UNSPECIFIED if v is None else v

        # Special: dynamic_planning bool → None (disabled) or UNSPECIFIED
        # True (the build_parser default) → UNSPECIFIED (use library default)
        # False → explicitly disable (None)
        # absent / None → UNSPECIFIED
        dyn = getattr(args, "dynamic_planning", None)
        dynamic_planning: Any = None if dyn is False else UNSPECIFIED

        # Special: stream_policy "auto" or absent → UNSPECIFIED
        sp = getattr(args, "stream_policy", None)
        cuda_stream_policy: Any = UNSPECIFIED if (sp is None or sp == "auto") else sp

        # target_partition_size: canonical dest from add_cli_args; fall back to
        # "blocksize" for legacy benchmark scripts that predate this module.
        target_partition_size = (
            _get("target_partition_size")
            if "target_partition_size" in vars(args)
            else _get("blocksize")
        )

        return cls(
            num_streaming_threads=_get("num_streaming_threads"),
            num_streams=_get("num_streams"),
            log=_get("rapidsmpf_log"),  # renamed: dest rapidsmpf_log → log
            statistics=_get("rapidsmpf_statistics"),  # renamed
            memory_reserve_timeout=_get("memory_reserve_timeout"),
            allow_overbooking_by_default=_get("allow_overbooking_by_default"),
            pinned_memory=_get("pinned_memory"),
            pinned_initial_pool_size=_get("pinned_initial_pool_size"),
            spill_device_limit=_get("spill_device_limit"),
            periodic_spill_check=_get("periodic_spill_check"),
            num_py_executors=_get("num_py_executors"),
            fallback_mode=_get("fallback_mode"),
            max_rows_per_partition=_get("max_rows_per_partition"),
            broadcast_join_limit=_get("broadcast_join_limit"),
            target_partition_size=target_partition_size,
            dynamic_planning=dynamic_planning,
            unique_fraction=_get("unique_fraction"),
            raise_on_fail=_get("raise_on_fail"),
            parquet_options=_get("parquet_options"),
            memory_resource_config=_get("memory_resource_config"),
            cuda_stream_policy=cuda_stream_policy,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """
        Register :class:`StreamingOptions`-specific CLI arguments on *parser*.

        Helpful to implement all available options as CLI arguments.

        Parameters
        ----------
        parser
            The parser to augment in-place.

        Examples
        --------
        >>> parser = argparse.ArgumentParser()
        >>> StreamingOptions.add_cli_args(parser)
        >>> opts = StreamingOptions.from_argparse(parser.parse_args([]))
        """
        g = parser.add_argument_group("Streaming and RapidsMPF Options")
        g.add_argument(
            "--num-streaming-threads",
            dest="num_streaming_threads",
            default=None,
            type=int,
            help=textwrap.dedent("""\
                Number of threads used to execute coroutines. Must be > 0.
                Env: RAPIDSMPF_NUM_STREAMING_THREADS. Built-in default: 1."""),
        )
        g.add_argument(
            "--num-streams",
            dest="num_streams",
            default=None,
            type=int,
            help=textwrap.dedent("""\
                Number of CUDA streams used by RapidsMPF for concurrent GPU execution.
                Env: RAPIDSMPF_NUM_STREAMS. Built-in default: 16."""),
        )
        g.add_argument(
            "--rapidsmpf-log",
            dest="rapidsmpf_log",
            default=None,
            type=str,
            choices=["NONE", "PRINT", "WARN", "INFO", "DEBUG", "TRACE"],
            help=textwrap.dedent("""\
                RapidsMPF log verbosity level.
                Env: RAPIDSMPF_LOG. Built-in default: WARN."""),
        )
        g.add_argument(
            "--rapidsmpf-statistics",
            dest="rapidsmpf_statistics",
            default=None,
            action=argparse.BooleanOptionalAction,
            help=textwrap.dedent("""\
                Collect RapidsMPF performance metrics.
                Env: RAPIDSMPF_STATISTICS. Built-in default: false."""),
        )
        g.add_argument(
            "--memory-reserve-timeout",
            dest="memory_reserve_timeout",
            default=None,
            type=str,
            help=textwrap.dedent("""\
                Global timeout for memory reservation requests (e.g. "100ms").
                Env: RAPIDSMPF_MEMORY_RESERVE_TIMEOUT. Built-in default: 100ms."""),
        )
        g.add_argument(
            "--allow-overbooking",
            dest="allow_overbooking_by_default",
            default=None,
            action=argparse.BooleanOptionalAction,
            help=textwrap.dedent("""\
                Allow memory overallocation by default in high-level reservation APIs.
                Env: RAPIDSMPF_ALLOW_OVERBOOKING_BY_DEFAULT. Built-in default: true."""),
        )
        g.add_argument(
            "--pinned-memory",
            dest="pinned_memory",
            default=None,
            action=argparse.BooleanOptionalAction,
            help=textwrap.dedent("""\
                Enable pinned host memory if available on the system.
                Env: RAPIDSMPF_PINNED_MEMORY. Built-in default: false."""),
        )
        g.add_argument(
            "--pinned-initial-pool-size",
            dest="pinned_initial_pool_size",
            default=None,
            type=int,
            help=textwrap.dedent("""\
                Starting allocation for the pinned memory pool in bytes.
                Env: RAPIDSMPF_PINNED_INITIAL_POOL_SIZE. Built-in default: 0."""),
        )
        g.add_argument(
            "--spill-device-limit",
            dest="spill_device_limit",
            default=None,
            type=str,
            help=textwrap.dedent("""\
                Soft upper limit on device memory usage before spilling,
                expressed as a percentage (e.g. "80%%") or byte count.
                Env: RAPIDSMPF_SPILL_DEVICE_LIMIT. Built-in default: 80%%."""),
        )
        g.add_argument(
            "--periodic-spill-check",
            dest="periodic_spill_check",
            default=None,
            type=str,
            help=textwrap.dedent("""\
                Interval between periodic spill checks (e.g. "1ms").
                Env: RAPIDSMPF_PERIODIC_SPILL_CHECK. Built-in default: 1ms."""),
        )
        g.add_argument(
            "--num-py-executors",
            dest="num_py_executors",
            default=None,
            type=int,
            help=textwrap.dedent("""\
                Max workers for the Python ThreadPoolExecutor inside RapidsMPF.
                Env: CUDF_POLARS__EXECUTOR__NUM_PY_EXECUTORS.
                Built-in default: 1."""),
        )
        g.add_argument(
            "--raise-on-fail",
            dest="raise_on_fail",
            default=None,
            action=argparse.BooleanOptionalAction,
            help=textwrap.dedent("""\
                Raise an exception instead of falling back to CPU when the GPU
                engine cannot execute a query.
                Built-in default: false."""),
        )
        g.add_argument(
            "--fallback-mode",
            dest="fallback_mode",
            default=None,
            type=str,
            choices=["warn", "raise", "silent"],
            help=textwrap.dedent("""\
                Behavior when a query node cannot run on the GPU.
                Env: CUDF_POLARS__EXECUTOR__FALLBACK_MODE. Built-in default: warn."""),
        )
        g.add_argument(
            "--max-rows-per-partition",
            dest="max_rows_per_partition",
            default=None,
            type=int,
            help=textwrap.dedent("""\
                Maximum number of rows per partition.
                Env: CUDF_POLARS__EXECUTOR__MAX_ROWS_PER_PARTITION.
                Built-in default: 1000000."""),
        )
        g.add_argument(
            "--broadcast-join-limit",
            dest="broadcast_join_limit",
            default=None,
            type=int,
            help=textwrap.dedent("""\
                Maximum number of partitions eligible for broadcast joins.
                Env: CUDF_POLARS__EXECUTOR__BROADCAST_JOIN_LIMIT. Built-in default: auto."""),
        )
        g.add_argument(
            "--target-partition-size",
            dest="target_partition_size",
            default=None,
            type=int,
            help=textwrap.dedent("""\
                Target IO partition size in bytes. 0 = auto.
                Env: CUDF_POLARS__EXECUTOR__TARGET_PARTITION_SIZE. Built-in default: auto."""),
        )
        g.add_argument(
            "--dynamic-planning",
            dest="dynamic_planning",
            default=None,
            action=argparse.BooleanOptionalAction,
            help=textwrap.dedent("""\
                Enable dynamic planning. Use --no-dynamic-planning to disable.
                Env: CUDF_POLARS__EXECUTOR__DYNAMIC_PLANNING. Built-in default: enabled."""),
        )
        g.add_argument(
            "--unique-fraction",
            dest="unique_fraction",
            default=None,
            type=json.loads,
            help=textwrap.dedent("""\
                Per-column uniqueness estimate as a JSON object (e.g. '{"col": 0.5}').
                Env: CUDF_POLARS__EXECUTOR__UNIQUE_FRACTION. Built-in default: {}."""),
        )
        g.add_argument(
            "--stream-policy",
            dest="stream_policy",
            default=None,
            type=str,
            choices=["auto", "default", "new", "pool"],
            help=textwrap.dedent("""\
                CUDA stream pool policy. "auto" defers to the built-in default.
                Env: CUDF_POLARS__CUDA_STREAM_POLICY. Built-in default: default."""),
        )
        g.add_argument(
            "--parquet-options",
            dest="parquet_options",
            default=None,
            type=json.loads,
            help=textwrap.dedent("""\
                Parquet configuration as a JSON object.
                Env: CUDF_POLARS__PARQUET_OPTIONS__*."""),
        )
        g.add_argument(
            "--memory-resource-config",
            dest="memory_resource_config",
            default=None,
            type=json.loads,
            help=textwrap.dedent("""\
                RMM memory resource configuration as a JSON object.
                Env: CUDF_POLARS__MEMORY_RESOURCE_CONFIG__*."""),
        )
