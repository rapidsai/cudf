# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for :class:`polars.GPUEngine` tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import polars as pl

    from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions


def is_streaming_engine(obj: Any) -> bool:
    """True when ``obj`` is a :class:`StreamingEngine`."""
    try:
        from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine
    except ImportError:
        return False
    return isinstance(obj, StreamingEngine)


def get_blocksize_mode(obj: pl.GPUEngine) -> Literal["default", "small"]:
    """
    Recover the blocksize mode an engine was configured with.

    Inspects ``max_rows_per_partition`` in the engine's executor options
    and returns ``"small"`` if it matches the value set by the
    ``blocksize_mode == "small"`` branch of ``streaming_engine_factory``,
    otherwise ``"default"``. Non-streaming engines have no blocksize mode
    and always return ``"default"``.
    """
    if not is_streaming_engine(obj):
        return "default"
    executor_options = obj.config.get("executor_options", {})
    return "small" if executor_options.get("max_rows_per_partition") == 4 else "default"


def create_streaming_options(
    blocksize_mode: Literal["default", "small"],
) -> StreamingOptions:
    """
    Return the :class:`StreamingOptions` used by tests for ``blocksize_mode``.

    ``"small"`` sets tiny partitions plus ``fallback_mode=SILENT`` so tests
    that fall back to CPU don't drown the suite in warnings.
    """
    from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
    from cudf_polars.utils.config import StreamingFallbackMode

    if blocksize_mode == "default":
        return StreamingOptions(
            max_rows_per_partition=50,
            dynamic_planning={},
            target_partition_size=1_000_000,
            raise_on_fail=True,
        )
    if blocksize_mode == "small":
        return StreamingOptions(
            max_rows_per_partition=4,
            dynamic_planning={},
            target_partition_size=10,
            raise_on_fail=True,
            fallback_mode=StreamingFallbackMode.SILENT,
        )
    raise ValueError(f"Unknown blocksize_mode: {blocksize_mode!r}")
