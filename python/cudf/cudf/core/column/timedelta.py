# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from __future__ import annotations

import datetime
from typing import Any, Optional, Sequence, cast

import numpy as np
import pandas as pd
import pyarrow as pa

import cudf
from cudf import _lib as libcudf
from cudf._typing import ColumnBinaryOperand, DatetimeLikeScalar, Dtype
from cudf.api.types import is_scalar, is_timedelta64_dtype
from cudf.core.buffer import Buffer, acquire_spill_lock
from cudf.core.column import ColumnBase, column, string
from cudf.utils.dtypes import np_to_pa_dtype
from cudf.utils.utils import _all_bools_with_nulls, _fillna_natwise

_dtype_to_format_conversion = {
    "timedelta64[ns]": "%D days %H:%M:%S",
    "timedelta64[us]": "%D days %H:%M:%S",
    "timedelta64[ms]": "%D days %H:%M:%S",
    "timedelta64[s]": "%D days %H:%M:%S",
}

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
        dtype: Dtype,
        size: Optional[int] = None,  # TODO: make non-optional
        mask: Optional[Buffer] = None,
        offset: int = 0,
        null_count: Optional[int] = None,
    ):
        dtype = cudf.dtype(dtype)

        if data.size % dtype.itemsize:
            raise ValueError("Buffer size must be divisible by element size")
        if size is None:
            size = data.size // dtype.itemsize
            size = size - offset
        super().__init__(
            data,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
        )

        if self.dtype.type is not np.timedelta64:
            raise TypeError(f"{self.dtype} is not a supported duration type")

        self._time_unit, _ = np.datetime_data(self.dtype)

    def __contains__(self, item: DatetimeLikeScalar) -> bool:
        try:
            item = np.timedelta64(item, self._time_unit)
        except ValueError:
            # If item cannot be converted to duration type
            # np.timedelta64 raises ValueError, hence `item`
            # cannot exist in `self`.
            return False
        return item.view("int64") in self.as_numerical

    @property
    def values(self):
        """
        Return a CuPy representation of the TimeDeltaColumn.
        """
        raise NotImplementedError(
            "TimeDelta Arrays is not yet implemented in cudf"
        )

    @acquire_spill_lock()
    def to_arrow(self) -> pa.Array:
        mask = None
        if self.nullable:
            mask = pa.py_buffer(
                self.mask_array_view(mode="read").copy_to_host()
            )
        data = pa.py_buffer(
            self.as_numerical.data_array_view(mode="read").copy_to_host()
        )
        pa_dtype = np_to_pa_dtype(self.dtype)
        return pa.Array.from_buffers(
            type=pa_dtype,
            length=len(self),
            buffers=[mask, data],
            null_count=self.null_count,
        )

    def to_pandas(
        self, index=None, nullable: bool = False, **kwargs
    ) -> pd.Series:
        # Workaround until following issue is fixed:
        # https://issues.apache.org/jira/browse/ARROW-9772

        # Pandas supports only `timedelta64[ns]`, hence the cast.
        pd_series = pd.Series(
            self.astype("timedelta64[ns]").fillna("NaT").values_host,
            copy=False,
        )

        if index is not None:
            pd_series.index = index

        return pd_series

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        reflect, op = self._check_reflected_op(op)
        other = self._wrap_binop_normalization(other)
        if other is NotImplemented:
            return NotImplemented

        this: ColumnBinaryOperand = self
        out_dtype = None

        if is_timedelta64_dtype(other.dtype):
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
                        other = other.value.astype(common_dtype).astype(
                            out_dtype
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
            elif op in {"__eq__", "NULL_EQUALS", "__ne__"}:
                if isinstance(other, ColumnBase) and not isinstance(
                    other, TimeDeltaColumn
                ):
                    result = _all_bools_with_nulls(
                        self, other, bool_fill_value=op == "__ne__"
                    )
                    if cudf.get_option("mode.pandas_compatible"):
                        result = result.fillna(op == "__ne__")
                    return result

        if out_dtype is None:
            return NotImplemented

        lhs, rhs = (other, this) if reflect else (this, other)

        result = libcudf.binaryop.binaryop(lhs, rhs, op, out_dtype)
        if cudf.get_option(
            "mode.pandas_compatible"
        ) and out_dtype == cudf.dtype(np.bool_):
            result = result.fillna(op == "__ne__")
        return result

    def normalize_binop_value(self, other) -> ColumnBinaryOperand:
        if isinstance(other, (ColumnBase, cudf.Scalar)):
            return other

        tz_error_msg = (
            "Cannot perform binary operation on timezone-naive columns"
            " and timezone-aware timestamps."
        )
        if isinstance(other, pd.Timestamp):
            if other.tz is not None:
                raise NotImplementedError(tz_error_msg)
            other = other.to_datetime64()
        elif isinstance(other, pd.Timedelta):
            other = other.to_timedelta64()
        elif isinstance(other, datetime.timedelta):
            other = np.timedelta64(other)
        elif isinstance(other, datetime.datetime) and other.tzinfo is not None:
            raise NotImplementedError(tz_error_msg)

        if isinstance(other, np.timedelta64):
            other_time_unit = cudf.utils.dtypes.get_time_unit(other)
            if np.isnat(other):
                return cudf.Scalar(None, dtype=self.dtype)

            if other_time_unit not in {"s", "ms", "ns", "us"}:
                common_dtype = "timedelta64[s]"
            else:
                common_dtype = determine_out_dtype(self.dtype, other.dtype)
            return cudf.Scalar(other.astype(common_dtype))
        elif np.isscalar(other):
            return cudf.Scalar(other)
        return NotImplemented

    @property
    def as_numerical(self) -> "cudf.core.column.NumericalColumn":
        return cast(
            "cudf.core.column.NumericalColumn",
            column.build_column(
                data=self.base_data,
                dtype=np.int64,
                mask=self.base_mask,
                offset=self.offset,
                size=self.size,
            ),
        )

    @property
    def time_unit(self) -> str:
        return self._time_unit

    def fillna(
        self,
        fill_value: Any = None,
        method: Optional[str] = None,
        dtype: Optional[Dtype] = None,
    ) -> TimeDeltaColumn:
        if fill_value is not None:
            if cudf.utils.utils._isnat(fill_value):
                return _fillna_natwise(self)
            col: ColumnBase = self
            if is_scalar(fill_value):
                if isinstance(fill_value, np.timedelta64):
                    dtype = determine_out_dtype(self.dtype, fill_value.dtype)
                    fill_value = fill_value.astype(dtype)
                    col = col.astype(dtype)
                if not isinstance(fill_value, cudf.Scalar):
                    fill_value = cudf.Scalar(fill_value, dtype=dtype)
            else:
                fill_value = column.as_column(fill_value, nan_as_null=False)
            return cast(TimeDeltaColumn, ColumnBase.fillna(col, fill_value))
        else:
            return super().fillna(method=method)

    def as_numerical_column(
        self, dtype: Dtype, **kwargs
    ) -> "cudf.core.column.NumericalColumn":
        return cast(
            "cudf.core.column.NumericalColumn", self.as_numerical.astype(dtype)
        )

    def as_datetime_column(
        self, dtype: Dtype, **kwargs
    ) -> "cudf.core.column.DatetimeColumn":
        raise TypeError(
            f"cannot astype a timedelta from {self.dtype} to {dtype}"
        )

    def as_string_column(
        self, dtype: Dtype, format=None, **kwargs
    ) -> "cudf.core.column.StringColumn":
        if format is None:
            format = _dtype_to_format_conversion.get(
                self.dtype.name, "%D days %H:%M:%S"
            )
        if len(self) > 0:
            return string._timedelta_to_str_typecast_functions[
                cudf.dtype(self.dtype)
            ](self, format=format)
        else:
            return cast(
                "cudf.core.column.StringColumn",
                column.column_empty(0, dtype="object", masked=False),
            )

    def as_timedelta_column(self, dtype: Dtype, **kwargs) -> TimeDeltaColumn:
        dtype = cudf.dtype(dtype)
        if dtype == self.dtype:
            return self
        return libcudf.unary.cast(self, dtype=dtype)

    def mean(self, skipna=None, dtype: Dtype = np.float64) -> pd.Timedelta:
        return pd.Timedelta(
            self.as_numerical.mean(skipna=skipna, dtype=dtype),
            unit=self.time_unit,
        )

    def median(self, skipna: Optional[bool] = None) -> pd.Timedelta:
        return pd.Timedelta(
            self.as_numerical.median(skipna=skipna), unit=self.time_unit
        )

    def isin(self, values: Sequence) -> ColumnBase:
        return cudf.core.tools.datetimes._isin_datetimelike(self, values)

    def quantile(
        self,
        q: np.ndarray,
        interpolation: str,
        exact: bool,
        return_scalar: bool,
    ) -> ColumnBase:
        result = self.as_numerical.quantile(
            q=q,
            interpolation=interpolation,
            exact=exact,
            return_scalar=return_scalar,
        )
        if return_scalar:
            return pd.Timedelta(result, unit=self.time_unit)
        return result.astype(self.dtype)

    def sum(
        self,
        skipna: Optional[bool] = None,
        min_count: int = 0,
        dtype: Optional[Dtype] = None,
    ) -> pd.Timedelta:
        return pd.Timedelta(
            # Since sum isn't overridden in Numerical[Base]Column, mypy only
            # sees the signature from Reducible (which doesn't have the extra
            # parameters from ColumnBase._reduce) so we have to ignore this.
            self.as_numerical.sum(  # type: ignore
                skipna=skipna, min_count=min_count, dtype=dtype
            ),
            unit=self.time_unit,
        )

    def std(
        self,
        skipna: Optional[bool] = None,
        min_count: int = 0,
        dtype: Dtype = np.float64,
        ddof: int = 1,
    ) -> pd.Timedelta:
        return pd.Timedelta(
            self.as_numerical.std(
                skipna=skipna, min_count=min_count, ddof=ddof, dtype=dtype
            ),
            unit=self.time_unit,
        )

    def components(self, index=None) -> "cudf.DataFrame":
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
        """  # noqa: E501

        return cudf.DataFrame(
            data={
                "days": self
                // cudf.Scalar(
                    np.timedelta64(_unit_to_nanoseconds_conversion["D"], "ns")
                ),
                "hours": (
                    self
                    % cudf.Scalar(
                        np.timedelta64(
                            _unit_to_nanoseconds_conversion["D"], "ns"
                        )
                    )
                )
                // cudf.Scalar(
                    np.timedelta64(_unit_to_nanoseconds_conversion["h"], "ns")
                ),
                "minutes": (
                    self
                    % cudf.Scalar(
                        np.timedelta64(
                            _unit_to_nanoseconds_conversion["h"], "ns"
                        )
                    )
                )
                // cudf.Scalar(
                    np.timedelta64(_unit_to_nanoseconds_conversion["m"], "ns")
                ),
                "seconds": (
                    self
                    % cudf.Scalar(
                        np.timedelta64(
                            _unit_to_nanoseconds_conversion["m"], "ns"
                        )
                    )
                )
                // cudf.Scalar(
                    np.timedelta64(_unit_to_nanoseconds_conversion["s"], "ns")
                ),
                "milliseconds": (
                    self
                    % cudf.Scalar(
                        np.timedelta64(
                            _unit_to_nanoseconds_conversion["s"], "ns"
                        )
                    )
                )
                // cudf.Scalar(
                    np.timedelta64(_unit_to_nanoseconds_conversion["ms"], "ns")
                ),
                "microseconds": (
                    self
                    % cudf.Scalar(
                        np.timedelta64(
                            _unit_to_nanoseconds_conversion["ms"], "ns"
                        )
                    )
                )
                // cudf.Scalar(
                    np.timedelta64(_unit_to_nanoseconds_conversion["us"], "ns")
                ),
                "nanoseconds": (
                    self
                    % cudf.Scalar(
                        np.timedelta64(
                            _unit_to_nanoseconds_conversion["us"], "ns"
                        )
                    )
                )
                // cudf.Scalar(
                    np.timedelta64(_unit_to_nanoseconds_conversion["ns"], "ns")
                ),
            },
            index=index,
        )

    @property
    def days(self) -> "cudf.core.column.NumericalColumn":
        """
        Number of days for each element.

        Returns
        -------
        NumericalColumn
        """
        return self // cudf.Scalar(
            np.timedelta64(_unit_to_nanoseconds_conversion["D"], "ns")
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
                np.timedelta64(_unit_to_nanoseconds_conversion["D"], "ns")
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
            self % np.timedelta64(_unit_to_nanoseconds_conversion["s"], "ns")
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
