# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""A datatype, preserving polars metadata."""

from __future__ import annotations

from functools import cache

from typing_extensions import assert_never

import polars as pl

import pylibcudf as plc

__all__ = ["DataType"]


@cache
def _from_polars(dtype: pl.DataType) -> plc.DataType:
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
    elif isinstance(dtype, pl.Decimal):
        return plc.DataType(plc.TypeId.DECIMAL128, scale=-dtype.scale)
    elif isinstance(dtype, pl.Null):
        # TODO: Hopefully
        return plc.DataType(plc.TypeId.EMPTY)
    elif isinstance(dtype, pl.List):
        # Recurse to catch unsupported inner types
        _ = _from_polars(dtype.inner)
        return plc.DataType(plc.TypeId.LIST)
    elif isinstance(dtype, pl.Struct):
        # Recurse to catch unsupported field types
        for field in dtype.fields:
            _ = _from_polars(field.dtype)
        return plc.DataType(plc.TypeId.STRUCT)
    else:
        raise NotImplementedError(f"{dtype=} conversion not supported")


class DataType:
    """A datatype, preserving polars metadata."""

    polars_type: pl.datatypes.DataType
    plc_type: plc.DataType

    def __init__(self, polars_dtype: pl.DataType) -> None:
        self.polars_type = polars_dtype
        self.plc_type = _from_polars(polars_dtype)

    def id(self) -> plc.TypeId:
        """The pylibcudf.TypeId of this DataType."""
        return self.plc_type.id()

    @property
    def children(self) -> list[DataType]:
        """The children types of this DataType."""
        # these type ignores are needed because the type checker doesn't
        # see that these equality checks passing imply a specific type for each child field.
        if self.plc_type.id() == plc.TypeId.STRUCT:
            return [DataType(field.dtype) for field in self.polars_type.fields]
        elif self.plc_type.id() == plc.TypeId.LIST:
            return [DataType(self.polars_type.inner)]
        return []

    def scale(self) -> int:
        """The scale of this DataType."""
        return self.plc_type.scale()

    @staticmethod
    def common_decimal_dtype(left: DataType, right: DataType) -> DataType:
        """Return a common decimal DataType for the two inputs."""
        if not (
            plc.traits.is_fixed_point(left.plc_type)
            and plc.traits.is_fixed_point(right.plc_type)
        ):
            raise ValueError("Both inputs required to be decimal types.")
        return DataType(pl.Decimal(38, abs(min(left.scale(), right.scale()))))

    def __eq__(self, other: object) -> bool:
        """Equality of DataTypes."""
        if not isinstance(other, DataType):
            return False
        return self.polars_type == other.polars_type

    def __hash__(self) -> int:
        """Hash of the DataType."""
        return hash(self.polars_type)

    def __repr__(self) -> str:
        """Representation of the DataType."""
        return f"<DataType(polars={self.polars_type}, plc={self.id()!r})>"
