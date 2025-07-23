# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""A datatype, preserving polars metadata."""

from __future__ import annotations

from functools import cache

import polars as pl

import pylibcudf as plc

__all__ = ["DataType"]


POLARS_TO_PLC_TYPE = {
    pl.Boolean: plc.TypeId.BOOL8,
    pl.Int8: plc.TypeId.INT8,
    pl.Int16: plc.TypeId.INT16,
    pl.Int32: plc.TypeId.INT32,
    pl.Int64: plc.TypeId.INT64,
    pl.UInt8: plc.TypeId.UINT8,
    pl.UInt16: plc.TypeId.UINT16,
    pl.UInt32: plc.TypeId.UINT32,
    pl.UInt64: plc.TypeId.UINT64,
    pl.Float32: plc.TypeId.FLOAT32,
    pl.Float64: plc.TypeId.FLOAT64,
    pl.Date: plc.TypeId.TIMESTAMP_DAYS,
    pl.String: plc.TypeId.STRING,
    pl.Null: plc.TypeId.EMPTY,
}

PLC_TO_POLARS_TYPE = {v: k for k, v in POLARS_TO_PLC_TYPE.items()}

TIME_UNIT_TO_PLC_DATETIME_TYPE = {
    "ms": plc.TypeId.TIMESTAMP_MILLISECONDS,
    "us": plc.TypeId.TIMESTAMP_MICROSECONDS,
    "ns": plc.TypeId.TIMESTAMP_NANOSECONDS,
}
PLC_DATETIME_TYPE_TO_TIME_UNIT = {
    v: k for k, v in TIME_UNIT_TO_PLC_DATETIME_TYPE.items()
}

TIME_UNIT_TO_PLC_DURATION_TYPE = {
    "ms": plc.TypeId.DURATION_MILLISECONDS,
    "us": plc.TypeId.DURATION_MICROSECONDS,
    "ns": plc.TypeId.DURATION_NANOSECONDS,
}
PLC_DURATION_TYPE_TO_TIME_UNIT = {
    v: k for k, v in TIME_UNIT_TO_PLC_DURATION_TYPE.items()
}


@cache
def _from_polars(dtype: pl.DataType) -> plc.DataType:
    if type(dtype) in POLARS_TO_PLC_TYPE:
        return plc.DataType(POLARS_TO_PLC_TYPE[type(dtype)])

    if isinstance(dtype, pl.Datetime):
        return plc.DataType(TIME_UNIT_TO_PLC_DATETIME_TYPE[dtype.time_unit])

    if isinstance(dtype, pl.Duration):
        return plc.DataType(TIME_UNIT_TO_PLC_DURATION_TYPE[dtype.time_unit])

    if isinstance(dtype, pl.Time):
        raise NotImplementedError("Time of day dtype not implemented")

    if isinstance(dtype, pl.List):
        _ = _from_polars(dtype.inner)
        return plc.DataType(plc.TypeId.LIST)

    if isinstance(dtype, pl.Struct):
        for field in dtype.fields:
            _ = _from_polars(field.dtype)
        return plc.DataType(plc.TypeId.STRUCT)

    raise NotImplementedError(f"{dtype=} conversion not supported")


@cache
def _to_polars(dtype: plc.DataType) -> pl.DataType:
    type_id = dtype.id()

    if type_id in PLC_TO_POLARS_TYPE:
        return PLC_TO_POLARS_TYPE[type_id]()

    if type_id in PLC_DATETIME_TYPE_TO_TIME_UNIT:
        return pl.Datetime(PLC_DATETIME_TYPE_TO_TIME_UNIT[type_id])

    if type_id in PLC_DURATION_TYPE_TO_TIME_UNIT:
        return pl.Duration(PLC_DURATION_TYPE_TO_TIME_UNIT[type_id])

    # TODO: STRUCT and LIST support
    raise NotImplementedError(f"{dtype=} conversion not supported")


class DataType:
    """A datatype, preserving polars metadata."""

    polars: pl.DataType | pl.Field
    plc: plc.DataType

    def __init__(self, dtype: pl.Field | pl.DataType | plc.DataType) -> None:
        if isinstance(dtype, (pl.DataType, pl.Field)):
            self.polars = dtype
            self.plc = _from_polars(dtype)
        elif isinstance(dtype, plc.DataType):
            self.polars = _to_polars(dtype)
            self.plc = dtype

    def id(self) -> plc.TypeId:
        """The pylibcudf.TypeId of this DataType."""
        return self.plc.id()

    @property
    def children(self) -> list[DataType]:
        """The children types of this DataType."""
        if self.plc.id() == plc.TypeId.STRUCT:
            return [DataType(field.dtype) for field in self.polars.fields]
        elif self.plc.id() == plc.TypeId.LIST:
            return [DataType(self.polars.inner)]
        return []

    def __eq__(self, other: object) -> bool:
        """Equality of DataTypes."""
        if not isinstance(other, DataType):
            return False
        return self.polars == other.polars

    def __hash__(self) -> int:
        """Hash of the DataType."""
        return hash(self.polars)

    def __repr__(self) -> str:
        """Representation of the DataType."""
        return f"<DataType(polars={self.polars}, plc={self.id()!r})>"
