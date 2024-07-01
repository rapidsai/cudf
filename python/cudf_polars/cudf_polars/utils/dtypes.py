# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Datatype utilities."""

from __future__ import annotations

from functools import cache

import pyarrow as pa
from typing_extensions import assert_never

import polars as pl

import cudf._lib.pylibcudf as plc

__all__ = ["from_polars", "downcast_arrow_lists", "have_compatible_resolution"]


TIMELIKE_TYPES: frozenset[plc.TypeId] = frozenset(
    [
        plc.TypeId.TIMESTAMP_MILLISECONDS,
        plc.TypeId.TIMESTAMP_MICROSECONDS,
        plc.TypeId.TIMESTAMP_NANOSECONDS,
        plc.TypeId.TIMESTAMP_DAYS,
        plc.TypeId.DURATION_MILLISECONDS,
        plc.TypeId.DURATION_MICROSECONDS,
        plc.TypeId.DURATION_NANOSECONDS,
    ]
)


def have_compatible_resolution(lid: plc.TypeId, rid: plc.TypeId):
    """
    Do two datetime typeids have matching resolution for a binop.

    Parameters
    ----------
    lid
       Left type id
    rid
       Right type id

    Returns
    -------
    True if resolutions are compatible, False otherwise.

    Notes
    -----
    Polars has different casting rules for combining
    datetimes/durations than libcudf, and while we don't encode the
    casting rules fully, just reject things we can't handle.

    Precondition for correctness: both lid and rid are timelike.
    """
    if lid == rid:
        return True
    # Timestamps are smaller than durations in the libcudf enum.
    lid, rid = sorted([lid, rid])
    if lid == plc.TypeId.TIMESTAMP_MILLISECONDS:
        return rid == plc.TypeId.DURATION_MILLISECONDS
    elif lid == plc.TypeId.TIMESTAMP_MICROSECONDS:
        return rid == plc.TypeId.DURATION_MICROSECONDS
    elif lid == plc.TypeId.TIMESTAMP_NANOSECONDS:
        return rid == plc.TypeId.DURATION_NANOSECONDS
    return False


def downcast_arrow_lists(typ: pa.DataType) -> pa.DataType:
    """
    Sanitize an arrow datatype from polars.

    Parameters
    ----------
    typ
        Arrow type to sanitize

    Returns
    -------
    Sanitized arrow type

    Notes
    -----
    As well as arrow ``ListType``s, polars can produce
    ``LargeListType``s and ``FixedSizeListType``s, these are not
    currently handled by libcudf, so we attempt to cast them all into
    normal ``ListType``s on the arrow side before consuming the arrow
    data.
    """
    if isinstance(typ, pa.LargeListType):
        return pa.list_(downcast_arrow_lists(typ.value_type))
    # We don't have to worry about diving into struct types for now
    # since those are always NotImplemented before we get here.
    assert not isinstance(typ, pa.StructType)
    return typ


@cache
def from_polars(dtype: pl.DataType) -> plc.DataType:
    """
    Convert a polars datatype to a pylibcudf one.

    Parameters
    ----------
    dtype
        Polars dtype to convert

    Returns
    -------
    Matching pylibcudf DataType object.

    Raises
    ------
    NotImplementedError
        For unsupported conversions.
    """
    if isinstance(dtype, pl.Boolean):
        return plc.DataType(plc.TypeId.BOOL8)
    elif isinstance(dtype, pl.Int8):
        return plc.DataType(plc.TypeId.INT8)
    elif isinstance(dtype, pl.Int16):
        return plc.DataType(plc.TypeId.INT16)
    elif isinstance(dtype, pl.Int32):
        return plc.DataType(plc.TypeId.INT32)
    elif isinstance(dtype, pl.Int64):
        return plc.DataType(plc.TypeId.INT64)
    if isinstance(dtype, pl.UInt8):
        return plc.DataType(plc.TypeId.UINT8)
    elif isinstance(dtype, pl.UInt16):
        return plc.DataType(plc.TypeId.UINT16)
    elif isinstance(dtype, pl.UInt32):
        return plc.DataType(plc.TypeId.UINT32)
    elif isinstance(dtype, pl.UInt64):
        return plc.DataType(plc.TypeId.UINT64)
    elif isinstance(dtype, pl.Float32):
        return plc.DataType(plc.TypeId.FLOAT32)
    elif isinstance(dtype, pl.Float64):
        return plc.DataType(plc.TypeId.FLOAT64)
    elif isinstance(dtype, pl.Date):
        return plc.DataType(plc.TypeId.TIMESTAMP_DAYS)
    elif isinstance(dtype, pl.Time):
        raise NotImplementedError("Time of day dtype not implemented")
    elif isinstance(dtype, pl.Datetime):
        if dtype.time_zone is not None:
            raise NotImplementedError("Time zone support")
        if dtype.time_unit == "ms":
            return plc.DataType(plc.TypeId.TIMESTAMP_MILLISECONDS)
        elif dtype.time_unit == "us":
            return plc.DataType(plc.TypeId.TIMESTAMP_MICROSECONDS)
        elif dtype.time_unit == "ns":
            return plc.DataType(plc.TypeId.TIMESTAMP_NANOSECONDS)
        assert dtype.time_unit is not None  # pragma: no cover
        assert_never(dtype.time_unit)
    elif isinstance(dtype, pl.Duration):
        if dtype.time_unit == "ms":
            return plc.DataType(plc.TypeId.DURATION_MILLISECONDS)
        elif dtype.time_unit == "us":
            return plc.DataType(plc.TypeId.DURATION_MICROSECONDS)
        elif dtype.time_unit == "ns":
            return plc.DataType(plc.TypeId.DURATION_NANOSECONDS)
        assert dtype.time_unit is not None  # pragma: no cover
        assert_never(dtype.time_unit)
    elif isinstance(dtype, pl.String):
        return plc.DataType(plc.TypeId.STRING)
    elif isinstance(dtype, pl.Null):
        # TODO: Hopefully
        return plc.DataType(plc.TypeId.EMPTY)
    elif isinstance(dtype, pl.List):
        # TODO: This doesn't consider the value type.
        # Recurse to catch unsupported inner types
        _ = from_polars(dtype.inner)
        return plc.DataType(plc.TypeId.LIST)
    else:
        raise NotImplementedError(f"{dtype=} conversion not supported")
