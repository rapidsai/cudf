# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for rolling window aggregations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc

if TYPE_CHECKING:
    from cudf_polars.typing import ClosedInterval, Duration


__all__ = [
    "duration_to_int",
    "duration_to_scalar",
    "offsets_to_windows",
    "range_window_bounds",
]


def duration_to_int(
    dtype: plc.DataType,
    months: int,
    weeks: int,
    days: int,
    nanoseconds: int,
    parsed_int: bool,  # noqa: FBT001
    negative: bool,  # noqa: FBT001
) -> int:
    """
    Convert a polars duration value to an integer.

    Parameters
    ----------
    dtype
        The type of the column being added to.
    months
        Number of months
    weeks
        Number of weeks
    days
        Number of days
    nanoseconds
        Number of nanoseconds
    parsed_int
        Is this actually a representation of an integer, not a duration?
    negative
        Is this a negative duration?

    Returns
    -------
    int
        The total number of nanoseconds represented by this duration,
        or just an integer if `parsed_int` was true.

    Raises
    ------
    NotImplementedError
        For unsupported durations or datatypes.
    """
    if months != 0:
        raise NotImplementedError("Month durations in rolling windows")
    if parsed_int and (weeks != 0 or days != 0 or dtype.id() != plc.TypeId.INT64):
        raise NotImplementedError(
            "Invalid duration for parsed_int"
        )  # pragma: no cover; polars raises first
    elif not parsed_int and dtype.id() == plc.TypeId.INT64:
        raise pl.exceptions.InvalidOperationError("Duration must be a parsed integer")
    value = nanoseconds + 24 * 60 * 60 * 10**9 * (days + 7 * weeks)
    return -value if negative else value


def duration_to_scalar(dtype: plc.DataType, value: int) -> plc.Scalar:
    """
    Convert a raw polars duration value to a pylibcudf scalar.

    Parameters
    ----------
    dtype
        The type of the column being added to.
    value
        The raw value as in integer. If `dtype` represents a timestamp
        type, this should be in nanoseconds.

    Returns
    -------
    pylibcudf.Scalar
        With datatype matching the provided dtype.

    Raises
    ------
    NotImplementedError
        For unsupported durations or datatypes.
    """
    tid = dtype.id()
    if tid == plc.TypeId.INT64:
        return plc.Scalar.from_py(value, dtype)
    elif tid == plc.TypeId.TIMESTAMP_NANOSECONDS:
        return plc.Scalar.from_py(value, plc.DataType(plc.TypeId.DURATION_NANOSECONDS))
    elif tid == plc.TypeId.TIMESTAMP_MICROSECONDS:
        return plc.Scalar.from_py(
            value // 1000, plc.DataType(plc.TypeId.DURATION_MICROSECONDS)
        )
    elif tid == plc.TypeId.TIMESTAMP_MILLISECONDS:
        return plc.Scalar.from_py(
            value // 1_000_000, plc.DataType(plc.TypeId.DURATION_MILLISECONDS)
        )
    else:
        raise NotImplementedError("Unsupported data type in rolling window offset")


def offsets_to_windows(
    dtype: plc.DataType,
    offset: Duration,
    period: Duration,
) -> tuple[plc.Scalar, plc.Scalar]:
    """
    Convert polars offset/period pair to preceding/following windows.

    Parameters
    ----------
    dtype
        Datatype of column defining windows
    offset
        Offset duration
    period
        Period of window

    Returns
    -------
    tuple of preceding and following windows as pylibcudf scalars.
    """
    offset_i = duration_to_int(dtype, *offset)
    period_i = duration_to_int(dtype, *period)
    # Polars uses current_row + offset, ..., current_row + offset + period
    # Libcudf uses current_row - preceding, ..., current_row + following
    return duration_to_scalar(dtype, -offset_i), duration_to_scalar(
        dtype, offset_i + period_i
    )


def range_window_bounds(
    preceding: plc.Scalar, following: plc.Scalar, closed_window: ClosedInterval
) -> tuple[plc.rolling.RangeWindowType, plc.rolling.RangeWindowType]:
    """
    Convert preceding and following scalars to range window specs.

    Parameters
    ----------
    preceding
        The preceding window scalar.
    following
        The following window scalar.
    closed_window
        How the window interval endpoints are treated.

    Returns
    -------
    tuple
        Of preceding and following range window types.
    """
    if closed_window == "both":
        return (
            plc.rolling.BoundedClosed(preceding),
            plc.rolling.BoundedClosed(following),
        )
    elif closed_window == "left":
        return (
            plc.rolling.BoundedClosed(preceding),
            plc.rolling.BoundedOpen(following),
        )
    elif closed_window == "right":
        return (
            plc.rolling.BoundedOpen(preceding),
            plc.rolling.BoundedClosed(following),
        )
    else:
        return (
            plc.rolling.BoundedOpen(preceding),
            plc.rolling.BoundedOpen(following),
        )
