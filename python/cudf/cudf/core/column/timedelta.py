# Copyright (c) 2020-2025, NVIDIA CORPORATION.

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf.core._internals import binaryop
from cudf.core.buffer import Buffer, acquire_spill_lock
from cudf.core.column.column import ColumnBase, as_column
from cudf.core.column.temporal_base import TemporalBaseColumn
from cudf.utils.dtypes import (
    cudf_dtype_from_pa_type,
    cudf_dtype_to_pa_type,
    find_common_type,
)
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.temporal import unit_to_nanoseconds_conversion

if TYPE_CHECKING:
    from cudf._typing import (
        ColumnBinaryOperand,
        DatetimeLikeScalar,
    )
    from cudf.core.column.numerical import NumericalColumn
    from cudf.core.column.string import StringColumn


@functools.cache
def get_np_td_unit_conversion(
    reso: str, dtype: None | np.dtype
) -> np.timedelta64:
    td = np.timedelta64(unit_to_nanoseconds_conversion[reso], "ns")
    if dtype is not None:
        return td.astype(dtype)
    return td


class TimeDeltaColumn(TemporalBaseColumn):
    """
    Parameters
    ----------
    data : Buffer
        The Timedelta values
    dtype : np.dtype
        The data type
    size : int
        Size of memory allocation.
    mask : Buffer; optional
        The validity mask
    offset : int
        Data offset
    null_count : int, optional
        The number of null values.
        If None, it is calculated automatically.
    """

    _NP_SCALAR = np.timedelta64
    _PD_SCALAR = pd.Timedelta
    _VALID_BINARY_OPERATIONS = {
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__add__",
        "__sub__",
        "__mul__",
        "__mod__",
        "__truediv__",
        "__floordiv__",
        "__radd__",
        "__rsub__",
        "__rmul__",
        "__rmod__",
        "__rtruediv__",
        "__rfloordiv__",
    }

    def __init__(
        self,
        data: Buffer,
        size: int | None,
        dtype: np.dtype,
        mask: Buffer | None = None,
        offset: int = 0,
        null_count: int | None = None,
        children: tuple = (),
    ):
        if cudf.get_option("mode.pandas_compatible"):
            if not dtype.kind == "m":
                raise ValueError("dtype must be a timedelta numpy dtype.")
        elif not (isinstance(dtype, np.dtype) and dtype.kind == "m"):
            raise ValueError("dtype must be a timedelta numpy dtype.")
        super().__init__(
            data=data,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
        )

    def _clear_cache(self) -> None:
        super()._clear_cache()
        attrs = (
            "days",
            "seconds",
            "microseconds",
            "nanoseconds",
            "time_unit",
        )
        for attr in attrs:
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def __contains__(self, item: DatetimeLikeScalar) -> bool:
        try:
            item = self._NP_SCALAR(item, self.time_unit)
        except ValueError:
            # If item cannot be converted to duration type
            # np.timedelta64 raises ValueError, hence `item`
            # cannot exist in `self`.
            return False
        return super().__contains__(item.to_numpy())

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        reflect, op = self._check_reflected_op(op)
        other = self._normalize_binop_operand(other)
        if other is NotImplemented:
            return NotImplemented

        this: ColumnBinaryOperand = self
        out_dtype = None
        other_cudf_dtype = (
            cudf_dtype_from_pa_type(other.type)
            if isinstance(other, pa.Scalar)
            else other.dtype
        )

        if other_cudf_dtype.kind == "m":
            # TODO: pandas will allow these operators to work but return false
            # when comparing to non-timedelta dtypes. We should do the same.
            if op in {
                "__eq__",
                "__ne__",
                "__lt__",
                "__gt__",
                "__le__",
                "__ge__",
                "NULL_EQUALS",
                "NULL_NOT_EQUALS",
            }:
                out_dtype = np.dtype(np.bool_)
            elif op == "__mod__":
                out_dtype = find_common_type((self.dtype, other_cudf_dtype))
            elif op in {"__truediv__", "__floordiv__"}:
                common_dtype = find_common_type((self.dtype, other_cudf_dtype))
                out_dtype = (
                    np.dtype(np.float64)
                    if op == "__truediv__"
                    else self._UNDERLYING_DTYPE
                )
                this = self.astype(common_dtype).astype(out_dtype)
                if isinstance(other, pa.Scalar):
                    if other.is_valid:
                        # pyarrow.cast doesn't support casting duration to float
                        # so go through numpy
                        other_np = pa.array([other]).to_numpy(
                            zero_copy_only=False
                        )
                        other_np = other_np.astype(common_dtype).astype(
                            out_dtype
                        )
                        other = pa.array(other_np)[0]
                    else:
                        other = pa.scalar(
                            None, type=cudf_dtype_to_pa_type(out_dtype)
                        )
                else:
                    other = other.astype(common_dtype).astype(out_dtype)
            elif op in {"__add__", "__sub__"}:
                out_dtype = find_common_type((self.dtype, other_cudf_dtype))
        elif other_cudf_dtype.kind in {"f", "i", "u"}:
            if op in {"__mul__", "__mod__", "__truediv__", "__floordiv__"}:
                out_dtype = self.dtype
            elif op in {"__eq__", "__ne__", "NULL_EQUALS", "NULL_NOT_EQUALS"}:
                if isinstance(other, ColumnBase) and not isinstance(
                    other, TimeDeltaColumn
                ):
                    fill_value = op in ("__ne__", "NULL_NOT_EQUALS")
                    result = self._all_bools_with_nulls(
                        other,
                        bool_fill_value=fill_value,
                    )
                    if cudf.get_option("mode.pandas_compatible"):
                        result = result.fillna(fill_value)
                    return result

        if out_dtype is None:
            return NotImplemented
        elif isinstance(other, pa.Scalar):
            other = pa_scalar_to_plc_scalar(other)

        lhs, rhs = (other, this) if reflect else (this, other)

        result = binaryop.binaryop(lhs, rhs, op, out_dtype)
        if cudf.get_option("mode.pandas_compatible") and out_dtype.kind == "b":
            result = result.fillna(op == "__ne__")
        return result

    def total_seconds(self) -> ColumnBase:
        conversion = unit_to_nanoseconds_conversion[self.time_unit] / 1e9
        # Typecast to decimal128 to avoid floating point precision issues
        # https://github.com/rapidsai/cudf/issues/17664
        return (
            (self.astype(self._UNDERLYING_DTYPE) * conversion)
            .astype(
                cudf.Decimal128Dtype(cudf.Decimal128Dtype.MAX_PRECISION, 9)
            )
            .round(decimals=abs(int(math.log10(conversion))))
            .astype(np.dtype(np.float64))
        )

    def as_datetime_column(self, dtype: np.dtype) -> None:  # type: ignore[override]
        raise TypeError(
            f"cannot astype a timedelta from {self.dtype} to {dtype}"
        )

    def strftime(self, format: str) -> StringColumn:
        if len(self) == 0:
            return super().strftime(format)
        else:
            with acquire_spill_lock():
                return type(self).from_pylibcudf(  # type: ignore[return-value]
                    plc.strings.convert.convert_durations.from_durations(
                        self.to_pylibcudf(mode="read"), format
                    )
                )

    def as_string_column(self, dtype) -> StringColumn:
        return self.strftime("%D days %H:%M:%S")

    def as_timedelta_column(self, dtype: np.dtype) -> TimeDeltaColumn:
        if dtype == self.dtype:
            return self
        return self.cast(dtype=dtype)  # type: ignore[return-value]

    def sum(
        self,
        skipna: bool | None = None,
        min_count: int = 0,
    ) -> pd.Timedelta:
        return self._PD_SCALAR(
            # Since sum isn't overridden in Numerical[Base]Column, mypy only
            # sees the signature from Reducible (which doesn't have the extra
            # parameters from ColumnBase._reduce) so we have to ignore this.
            self.astype(self._UNDERLYING_DTYPE).sum(  # type: ignore
                skipna=skipna, min_count=min_count
            ),
            unit=self.time_unit,
        ).as_unit(self.time_unit)

    def components(self) -> dict[str, NumericalColumn]:
        """
        Return a Dataframe of the components of the Timedeltas.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> s = pd.Series(pd.to_timedelta(np.arange(5), unit='s'))
        >>> s = cudf.Series([12231312123, 1231231231, 1123236768712, 2135656,
        ...     3244334234], dtype='timedelta64[ms]')
        >>> s
        0      141 days 13:35:12.123
        1       14 days 06:00:31.231
        2    13000 days 10:12:48.712
        3        0 days 00:35:35.656
        4       37 days 13:12:14.234
        dtype: timedelta64[ms]
        >>> s.dt.components
            days  hours  minutes  seconds  milliseconds  microseconds  nanoseconds
        0    141     13       35       12           123             0            0
        1     14      6        0       31           231             0            0
        2  13000     10       12       48           712             0            0
        3      0      0       35       35           656             0            0
        4     37     13       12       14           234             0            0
        """
        date_meta = {
            "hours": ["D", "h"],
            "minutes": ["h", "m"],
            "seconds": ["m", "s"],
            "milliseconds": ["s", "ms"],
            "microseconds": ["ms", "us"],
            "nanoseconds": ["us", "ns"],
        }
        data = {"days": self.days}
        reached_self_unit = False
        for result_key, (mod_unit, div_unit) in date_meta.items():
            if not reached_self_unit:
                res_col = (
                    self % get_np_td_unit_conversion(mod_unit, self.dtype)
                ) // get_np_td_unit_conversion(div_unit, self.dtype)
                reached_self_unit = self.time_unit == div_unit
            else:
                res_col = as_column(
                    0, length=len(self), dtype=self._UNDERLYING_DTYPE
                )
                if self.nullable:
                    res_col = res_col.set_mask(self.mask)
            data[result_key] = res_col
        return data

    @functools.cached_property
    def days(self) -> NumericalColumn:
        """
        Number of days for each element.

        Returns
        -------
        NumericalColumn
        """
        return self // get_np_td_unit_conversion("D", self.dtype)

    @functools.cached_property
    def seconds(self) -> NumericalColumn:
        """
        Number of seconds (>= 0 and less than 1 day).

        Returns
        -------
        NumericalColumn
        """
        # This property must return the number of seconds (>= 0 and
        # less than 1 day) for each element, hence first performing
        # mod operation to remove the number of days and then performing
        # division operation to extract the number of seconds.

        return (
            self % get_np_td_unit_conversion("D", self.dtype)
        ) // get_np_td_unit_conversion("s", None)

    @functools.cached_property
    def microseconds(self) -> NumericalColumn:
        """
        Number of microseconds (>= 0 and less than 1 second).

        Returns
        -------
        NumericalColumn
        """
        # This property must return the number of microseconds (>= 0 and
        # less than 1 second) for each element, hence first performing
        # mod operation to remove the number of seconds and then performing
        # division operation to extract the number of microseconds.

        return (
            self % get_np_td_unit_conversion("s", self.dtype)
        ) // get_np_td_unit_conversion("us", None)

    @functools.cached_property
    def nanoseconds(self) -> NumericalColumn:
        """
        Return the number of nanoseconds (n), where 0 <= n < 1 microsecond.

        Returns
        -------
        NumericalColumn
        """
        # This property must return the number of nanoseconds (>= 0 and
        # less than 1 microsecond) for each element, hence first performing
        # mod operation to remove the number of microseconds and then
        # performing division operation to extract the number
        # of nanoseconds.

        if self.time_unit != "ns":
            res_col = as_column(
                0, length=len(self), dtype=self._UNDERLYING_DTYPE
            )
            if self.nullable:
                res_col = res_col.set_mask(self.mask)
            return cast("cudf.core.column.NumericalColumn", res_col)
        return (
            self % get_np_td_unit_conversion("us", None)
        ) // get_np_td_unit_conversion("ns", None)
