# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from __future__ import annotations

import datetime
import functools
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pyarrow as pa

import cudf
import cudf.core.column.column as column
import cudf.core.column.string as string
from cudf.api.types import is_scalar
from cudf.core._internals import binaryop, unary
from cudf.core.buffer import Buffer, acquire_spill_lock
from cudf.core.column.column import ColumnBase
from cudf.utils.dtypes import np_to_pa_dtype
from cudf.utils.utils import (
    _all_bools_with_nulls,
    _datetime_timedelta_find_and_replace,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cudf._typing import ColumnBinaryOperand, DatetimeLikeScalar, Dtype

_unit_to_nanoseconds_conversion = {
    "ns": 1,
    "us": 1_000,
    "ms": 1_000_000,
    "s": 1_000_000_000,
    "m": 60_000_000_000,
    "h": 3_600_000_000_000,
    "D": 86_400_000_000_000,
}


class TimeDeltaColumn(ColumnBase):
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
        if not isinstance(data, Buffer):
            raise ValueError("data must be a Buffer.")
        if not (isinstance(dtype, np.dtype) and dtype.kind == "m"):
            raise ValueError("dtype must be a timedelta numpy dtype.")

        if data.size % dtype.itemsize:
            raise ValueError("Buffer size must be divisible by element size")
        if size is None:
            size = data.size // dtype.itemsize
            size = size - offset
        if len(children) != 0:
            raise ValueError("TimeDeltaColumn must have no children.")
        super().__init__(
            data=data,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
        )

    def __contains__(self, item: DatetimeLikeScalar) -> bool:
        try:
            item = np.timedelta64(item, self.time_unit)
        except ValueError:
            # If item cannot be converted to duration type
            # np.timedelta64 raises ValueError, hence `item`
            # cannot exist in `self`.
            return False
        return item.view("int64") in cast(
            "cudf.core.column.NumericalColumn", self.astype("int64")
        )

    @property
    def values(self):
        """
        Return a CuPy representation of the TimeDeltaColumn.
        """
        raise NotImplementedError(
            "TimeDelta Arrays is not yet implemented in cudf"
        )

    def element_indexing(self, index: int):
        result = super().element_indexing(index)
        if cudf.get_option("mode.pandas_compatible"):
            return pd.Timedelta(result)
        return result

    @acquire_spill_lock()
    def to_arrow(self) -> pa.Array:
        mask = None
        if self.nullable:
            mask = pa.py_buffer(
                self.mask_array_view(mode="read").copy_to_host()
            )
        data = pa.py_buffer(
            self.astype("int64").data_array_view(mode="read").copy_to_host()
        )
        pa_dtype = np_to_pa_dtype(self.dtype)
        return pa.Array.from_buffers(
            type=pa_dtype,
            length=len(self),
            buffers=[mask, data],
            null_count=self.null_count,
        )

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        reflect, op = self._check_reflected_op(op)
        other = self._wrap_binop_normalization(other)
        if other is NotImplemented:
            return NotImplemented

        this: ColumnBinaryOperand = self
        out_dtype = None

        if other.dtype.kind == "m":
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
                out_dtype = cudf.dtype(np.bool_)
            elif op == "__mod__":
                out_dtype = determine_out_dtype(self.dtype, other.dtype)
            elif op in {"__truediv__", "__floordiv__"}:
                common_dtype = determine_out_dtype(self.dtype, other.dtype)
                out_dtype = np.float64 if op == "__truediv__" else np.int64
                this = self.astype(common_dtype).astype(out_dtype)
                if isinstance(other, cudf.Scalar):
                    if other.is_valid():
                        other = cudf.Scalar(
                            other.value.astype(common_dtype).astype(out_dtype)
                        )
                    else:
                        other = cudf.Scalar(None, out_dtype)
                else:
                    other = other.astype(common_dtype).astype(out_dtype)
            elif op in {"__add__", "__sub__"}:
                out_dtype = determine_out_dtype(self.dtype, other.dtype)
        elif other.dtype.kind in {"f", "i", "u"}:
            if op in {"__mul__", "__mod__", "__truediv__", "__floordiv__"}:
                out_dtype = self.dtype
            elif op in {"__eq__", "__ne__", "NULL_EQUALS", "NULL_NOT_EQUALS"}:
                if isinstance(other, ColumnBase) and not isinstance(
                    other, TimeDeltaColumn
                ):
                    fill_value = op in ("__ne__", "NULL_NOT_EQUALS")
                    result = _all_bools_with_nulls(
                        self,
                        other,
                        bool_fill_value=fill_value,
                    )
                    if cudf.get_option("mode.pandas_compatible"):
                        result = result.fillna(fill_value)
                    return result

        if out_dtype is None:
            return NotImplemented

        lhs, rhs = (other, this) if reflect else (this, other)

        result = binaryop.binaryop(lhs, rhs, op, out_dtype)
        if cudf.get_option("mode.pandas_compatible") and out_dtype.kind == "b":
            result = result.fillna(op == "__ne__")
        return result

    def normalize_binop_value(self, other) -> ColumnBinaryOperand:
        if isinstance(other, (ColumnBase, cudf.Scalar)):
            return other

        tz_error_msg = (
            "Cannot perform binary operation on timezone-naive columns"
            " and timezone-aware timestamps."
        )
        if isinstance(other, datetime.datetime):
            if other.tzinfo is not None:
                raise NotImplementedError(tz_error_msg)
            other = pd.Timestamp(other).to_datetime64()
        elif isinstance(other, datetime.timedelta):
            other = pd.Timedelta(other).to_timedelta64()

        if isinstance(other, np.timedelta64):
            other_time_unit = cudf.utils.dtypes.get_time_unit(other)
            if np.isnat(other):
                return cudf.Scalar(
                    None,
                    dtype="timedelta64[ns]"
                    if other_time_unit not in {"s", "ms", "ns", "us"}
                    else self.dtype,
                )

            if other_time_unit not in {"s", "ms", "ns", "us"}:
                common_dtype = "timedelta64[s]"
            else:
                common_dtype = determine_out_dtype(self.dtype, other.dtype)
            return cudf.Scalar(other.astype(common_dtype))
        elif is_scalar(other):
            return cudf.Scalar(other)
        return NotImplemented

    @functools.cached_property
    def time_unit(self) -> str:
        return np.datetime_data(self.dtype)[0]

    def total_seconds(self) -> ColumnBase:
        raise NotImplementedError("total_seconds is currently not implemented")

    def ceil(self, freq: str) -> ColumnBase:
        raise NotImplementedError("ceil is currently not implemented")

    def floor(self, freq: str) -> ColumnBase:
        raise NotImplementedError("floor is currently not implemented")

    def round(self, freq: str) -> ColumnBase:
        raise NotImplementedError("round is currently not implemented")

    def as_numerical_column(
        self, dtype: Dtype
    ) -> cudf.core.column.NumericalColumn:
        col = cudf.core.column.NumericalColumn(
            data=self.base_data,  # type: ignore[arg-type]
            dtype=np.dtype(np.int64),
            mask=self.base_mask,
            offset=self.offset,
            size=self.size,
        )
        return cast("cudf.core.column.NumericalColumn", col.astype(dtype))

    def as_datetime_column(self, dtype: Dtype) -> None:  # type: ignore[override]
        raise TypeError(
            f"cannot astype a timedelta from {self.dtype} to {dtype}"
        )

    def strftime(self, format: str) -> cudf.core.column.StringColumn:
        if len(self) == 0:
            return cast(
                cudf.core.column.StringColumn,
                column.column_empty(0, dtype="object", masked=False),
            )
        else:
            return string._timedelta_to_str_typecast_functions[self.dtype](
                self, format=format
            )

    def as_string_column(self) -> cudf.core.column.StringColumn:
        return self.strftime("%D days %H:%M:%S")

    def as_timedelta_column(self, dtype: Dtype) -> TimeDeltaColumn:
        if dtype == self.dtype:
            return self
        return unary.cast(self, dtype=dtype)  # type: ignore[return-value]

    def find_and_replace(
        self,
        to_replace: ColumnBase,
        replacement: ColumnBase,
        all_nan: bool = False,
    ) -> TimeDeltaColumn:
        return cast(
            TimeDeltaColumn,
            _datetime_timedelta_find_and_replace(
                original_column=self,
                to_replace=to_replace,
                replacement=replacement,
                all_nan=all_nan,
            ),
        )

    def can_cast_safely(self, to_dtype: Dtype) -> bool:
        if to_dtype.kind == "m":  # type: ignore[union-attr]
            to_res, _ = np.datetime_data(to_dtype)
            self_res, _ = np.datetime_data(self.dtype)

            max_int = np.iinfo(np.int64).max

            max_dist = np.timedelta64(
                self.max().astype(np.int64, copy=False), self_res
            )
            min_dist = np.timedelta64(
                self.min().astype(np.int64, copy=False), self_res
            )

            self_delta_dtype = np.timedelta64(0, self_res).dtype

            if max_dist <= np.timedelta64(max_int, to_res).astype(
                self_delta_dtype
            ) and min_dist <= np.timedelta64(max_int, to_res).astype(
                self_delta_dtype
            ):
                return True
            else:
                return False
        elif to_dtype == cudf.dtype("int64") or to_dtype == cudf.dtype("O"):
            # can safely cast to representation, or string
            return True
        else:
            return False

    def mean(self, skipna=None) -> pd.Timedelta:
        return pd.Timedelta(
            cast(
                "cudf.core.column.NumericalColumn", self.astype("int64")
            ).mean(skipna=skipna),
            unit=self.time_unit,
        ).as_unit(self.time_unit)

    def median(self, skipna: bool | None = None) -> pd.Timedelta:
        return pd.Timedelta(
            cast(
                "cudf.core.column.NumericalColumn", self.astype("int64")
            ).median(skipna=skipna),
            unit=self.time_unit,
        ).as_unit(self.time_unit)

    def isin(self, values: Sequence) -> ColumnBase:
        return cudf.core.tools.datetimes._isin_datetimelike(self, values)

    def quantile(
        self,
        q: np.ndarray,
        interpolation: str,
        exact: bool,
        return_scalar: bool,
    ) -> ColumnBase:
        result = self.astype("int64").quantile(
            q=q,
            interpolation=interpolation,
            exact=exact,
            return_scalar=return_scalar,
        )
        if return_scalar:
            return pd.Timedelta(result, unit=self.time_unit).as_unit(
                self.time_unit
            )
        return result.astype(self.dtype)

    def sum(
        self,
        skipna: bool | None = None,
        min_count: int = 0,
        dtype: Dtype | None = None,
    ) -> pd.Timedelta:
        return pd.Timedelta(
            # Since sum isn't overridden in Numerical[Base]Column, mypy only
            # sees the signature from Reducible (which doesn't have the extra
            # parameters from ColumnBase._reduce) so we have to ignore this.
            self.astype("int64").sum(  # type: ignore
                skipna=skipna, min_count=min_count, dtype=dtype
            ),
            unit=self.time_unit,
        ).as_unit(self.time_unit)

    def std(
        self,
        skipna: bool | None = None,
        min_count: int = 0,
        ddof: int = 1,
    ) -> pd.Timedelta:
        return pd.Timedelta(
            cast("cudf.core.column.NumericalColumn", self.astype("int64")).std(
                skipna=skipna, min_count=min_count, ddof=ddof
            ),
            unit=self.time_unit,
        ).as_unit(self.time_unit)

    def cov(self, other: TimeDeltaColumn) -> float:
        if not isinstance(other, TimeDeltaColumn):
            raise TypeError(
                f"cannot perform cov with types {self.dtype}, {other.dtype}"
            )
        return cast(
            "cudf.core.column.NumericalColumn", self.astype("int64")
        ).cov(cast("cudf.core.column.NumericalColumn", other.astype("int64")))

    def corr(self, other: TimeDeltaColumn) -> float:
        if not isinstance(other, TimeDeltaColumn):
            raise TypeError(
                f"cannot perform corr with types {self.dtype}, {other.dtype}"
            )
        return cast(
            "cudf.core.column.NumericalColumn", self.astype("int64")
        ).corr(cast("cudf.core.column.NumericalColumn", other.astype("int64")))

    def components(self) -> dict[str, ColumnBase]:
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
            "seconds": ["m", "s"],
            "milliseconds": ["s", "ms"],
            "microseconds": ["ms", "us"],
            "nanoseconds": ["us", "ns"],
        }
        data = {
            "days": self
            // cudf.Scalar(
                np.timedelta64(
                    _unit_to_nanoseconds_conversion["D"], "ns"
                ).astype(self.dtype)
            ),
            "hours": (
                self
                % cudf.Scalar(
                    np.timedelta64(
                        _unit_to_nanoseconds_conversion["D"], "ns"
                    ).astype(self.dtype)
                )
            )
            // cudf.Scalar(
                np.timedelta64(
                    _unit_to_nanoseconds_conversion["h"], "ns"
                ).astype(self.dtype)
            ),
            "minutes": (
                self
                % cudf.Scalar(
                    np.timedelta64(
                        _unit_to_nanoseconds_conversion["h"], "ns"
                    ).astype(self.dtype)
                )
            )
            // cudf.Scalar(
                np.timedelta64(
                    _unit_to_nanoseconds_conversion["m"], "ns"
                ).astype(self.dtype)
            ),
        }
        keys_list = iter(date_meta.keys())
        for name in keys_list:
            value = date_meta[name]
            data[name] = (
                self
                % cudf.Scalar(
                    np.timedelta64(
                        _unit_to_nanoseconds_conversion[value[0]], "ns"
                    ).astype(self.dtype)
                )
            ) // cudf.Scalar(
                np.timedelta64(
                    _unit_to_nanoseconds_conversion[value[1]], "ns"
                ).astype(self.dtype)
            )
            if self.time_unit == value[1]:
                break

        for name in keys_list:
            res_col = column.as_column(0, length=len(self), dtype="int64")
            if self.nullable:
                res_col = res_col.set_mask(self.mask)
            data[name] = res_col
        return data

    @property
    def days(self) -> "cudf.core.column.NumericalColumn":
        """
        Number of days for each element.

        Returns
        -------
        NumericalColumn
        """
        return self // cudf.Scalar(
            np.timedelta64(_unit_to_nanoseconds_conversion["D"], "ns").astype(
                self.dtype
            )
        )

    @property
    def seconds(self) -> "cudf.core.column.NumericalColumn":
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
            self
            % cudf.Scalar(
                np.timedelta64(
                    _unit_to_nanoseconds_conversion["D"], "ns"
                ).astype(self.dtype)
            )
        ) // cudf.Scalar(
            np.timedelta64(_unit_to_nanoseconds_conversion["s"], "ns")
        )

    @property
    def microseconds(self) -> "cudf.core.column.NumericalColumn":
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
            self
            % np.timedelta64(
                _unit_to_nanoseconds_conversion["s"], "ns"
            ).astype(self.dtype)
        ) // cudf.Scalar(
            np.timedelta64(_unit_to_nanoseconds_conversion["us"], "ns")
        )

    @property
    def nanoseconds(self) -> "cudf.core.column.NumericalColumn":
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
            res_col = column.as_column(0, length=len(self), dtype="int64")
            if self.nullable:
                res_col = res_col.set_mask(self.mask)
            return cast("cudf.core.column.NumericalColumn", res_col)
        return (
            self
            % cudf.Scalar(
                np.timedelta64(_unit_to_nanoseconds_conversion["us"], "ns")
            )
        ) // cudf.Scalar(
            np.timedelta64(_unit_to_nanoseconds_conversion["ns"], "ns")
        )


def determine_out_dtype(lhs_dtype: Dtype, rhs_dtype: Dtype) -> Dtype:
    if np.can_cast(np.dtype(lhs_dtype), np.dtype(rhs_dtype)):
        return rhs_dtype
    elif np.can_cast(np.dtype(rhs_dtype), np.dtype(lhs_dtype)):
        return lhs_dtype
    else:
        raise TypeError(f"Cannot type-cast {lhs_dtype} and {rhs_dtype}")
