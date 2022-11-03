# Copyright (c) 2019-2022, NVIDIA CORPORATION.

from __future__ import annotations

import datetime
import locale
import re
from locale import nl_langinfo
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd

import cudf
from cudf import _lib as libcudf
from cudf._typing import (
    ColumnBinaryOperand,
    DatetimeLikeScalar,
    Dtype,
    DtypeObj,
    ScalarLike,
)
from cudf.api.types import is_datetime64_dtype, is_scalar, is_timedelta64_dtype
from cudf.core._compat import PANDAS_GE_120
from cudf.core.buffer import Buffer, cuda_array_interface_wrapper
from cudf.core.column import ColumnBase, as_column, column, string
from cudf.core.column.timedelta import _unit_to_nanoseconds_conversion
from cudf.utils.utils import _fillna_natwise

if PANDAS_GE_120:
    _guess_datetime_format = pd.core.tools.datetimes.guess_datetime_format
else:
    _guess_datetime_format = pd.core.tools.datetimes._guess_datetime_format

# nanoseconds per time_unit
_dtype_to_format_conversion = {
    "datetime64[ns]": "%Y-%m-%d %H:%M:%S.%9f",
    "datetime64[us]": "%Y-%m-%d %H:%M:%S.%6f",
    "datetime64[ms]": "%Y-%m-%d %H:%M:%S.%3f",
    "datetime64[s]": "%Y-%m-%d %H:%M:%S",
}

_DATETIME_SPECIAL_FORMATS = {
    "%b",
    "%B",
    "%A",
    "%a",
}

_DATETIME_NAMES = [
    nl_langinfo(locale.AM_STR),  # type: ignore
    nl_langinfo(locale.PM_STR),  # type: ignore
    nl_langinfo(locale.DAY_1),
    nl_langinfo(locale.DAY_2),
    nl_langinfo(locale.DAY_3),
    nl_langinfo(locale.DAY_4),
    nl_langinfo(locale.DAY_5),
    nl_langinfo(locale.DAY_6),
    nl_langinfo(locale.DAY_7),
    nl_langinfo(locale.ABDAY_1),
    nl_langinfo(locale.ABDAY_2),
    nl_langinfo(locale.ABDAY_3),
    nl_langinfo(locale.ABDAY_4),
    nl_langinfo(locale.ABDAY_5),
    nl_langinfo(locale.ABDAY_6),
    nl_langinfo(locale.ABDAY_7),
    nl_langinfo(locale.MON_1),
    nl_langinfo(locale.MON_2),
    nl_langinfo(locale.MON_3),
    nl_langinfo(locale.MON_4),
    nl_langinfo(locale.MON_5),
    nl_langinfo(locale.MON_6),
    nl_langinfo(locale.MON_7),
    nl_langinfo(locale.MON_8),
    nl_langinfo(locale.MON_9),
    nl_langinfo(locale.MON_10),
    nl_langinfo(locale.MON_11),
    nl_langinfo(locale.MON_12),
    nl_langinfo(locale.ABMON_1),
    nl_langinfo(locale.ABMON_2),
    nl_langinfo(locale.ABMON_3),
    nl_langinfo(locale.ABMON_4),
    nl_langinfo(locale.ABMON_5),
    nl_langinfo(locale.ABMON_6),
    nl_langinfo(locale.ABMON_7),
    nl_langinfo(locale.ABMON_8),
    nl_langinfo(locale.ABMON_9),
    nl_langinfo(locale.ABMON_10),
    nl_langinfo(locale.ABMON_11),
    nl_langinfo(locale.ABMON_12),
]


class DatetimeColumn(column.ColumnBase):
    """
    A Column implementation for Date-time types.

    Parameters
    ----------
    data : Buffer
        The datetime values
    dtype : np.dtype
        The data type
    mask : Buffer; optional
        The validity mask
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
        "__radd__",
        "__rsub__",
    }

    def __init__(
        self,
        data: Buffer,
        dtype: DtypeObj,
        mask: Buffer = None,
        size: int = None,  # TODO: make non-optional
        offset: int = 0,
        null_count: int = None,
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

        if not (self.dtype.type is np.datetime64):
            raise TypeError(f"{self.dtype} is not a supported datetime type")

        self._time_unit, _ = np.datetime_data(self.dtype)

    def __contains__(self, item: ScalarLike) -> bool:
        try:
            item_as_dt64 = np.datetime64(item, self._time_unit)
        except ValueError:
            # If item cannot be converted to datetime type
            # np.datetime64 raises ValueError, hence `item`
            # cannot exist in `self`.
            return False
        return item_as_dt64.astype("int64") in self.as_numerical

    @property
    def time_unit(self) -> str:
        return self._time_unit

    @property
    def year(self) -> ColumnBase:
        return self.get_dt_field("year")

    @property
    def month(self) -> ColumnBase:
        return self.get_dt_field("month")

    @property
    def day(self) -> ColumnBase:
        return self.get_dt_field("day")

    @property
    def hour(self) -> ColumnBase:
        return self.get_dt_field("hour")

    @property
    def minute(self) -> ColumnBase:
        return self.get_dt_field("minute")

    @property
    def second(self) -> ColumnBase:
        return self.get_dt_field("second")

    @property
    def weekday(self) -> ColumnBase:
        return self.get_dt_field("weekday")

    @property
    def dayofyear(self) -> ColumnBase:
        return self.get_dt_field("day_of_year")

    @property
    def day_of_year(self) -> ColumnBase:
        return self.get_dt_field("day_of_year")

    def to_pandas(
        self, index: pd.Index = None, nullable: bool = False, **kwargs
    ) -> "cudf.Series":
        # Workaround until following issue is fixed:
        # https://issues.apache.org/jira/browse/ARROW-9772

        # Pandas supports only `datetime64[ns]`, hence the cast.
        return pd.Series(
            self.astype("datetime64[ns]").fillna("NaT").values_host,
            copy=False,
            index=index,
        )

    @property
    def values(self):
        """
        Return a CuPy representation of the DateTimeColumn.
        """
        raise NotImplementedError(
            "DateTime Arrays is not yet implemented in cudf"
        )

    def get_dt_field(self, field: str) -> ColumnBase:
        return libcudf.datetime.extract_datetime_component(self, field)

    def ceil(self, freq: str) -> ColumnBase:
        return libcudf.datetime.ceil_datetime(self, freq)

    def floor(self, freq: str) -> ColumnBase:
        return libcudf.datetime.floor_datetime(self, freq)

    def round(self, freq: str) -> ColumnBase:
        return libcudf.datetime.round_datetime(self, freq)

    def normalize_binop_value(self, other: DatetimeLikeScalar) -> ScalarLike:
        if isinstance(other, (cudf.Scalar, ColumnBase, cudf.DateOffset)):
            return other

        if isinstance(other, datetime.datetime):
            other = np.datetime64(other)
        elif isinstance(other, datetime.timedelta):
            other = np.timedelta64(other)
        elif isinstance(other, pd.Timestamp):
            other = other.to_datetime64()
        elif isinstance(other, pd.Timedelta):
            other = other.to_timedelta64()

        if isinstance(other, np.datetime64):
            if np.isnat(other):
                return cudf.Scalar(None, dtype=self.dtype)

            other = other.astype(self.dtype)
            return cudf.Scalar(other)
        elif isinstance(other, np.timedelta64):
            other_time_unit = cudf.utils.dtypes.get_time_unit(other)

            if other_time_unit not in {"s", "ms", "ns", "us"}:
                other = other.astype("timedelta64[s]")

            if np.isnat(other):
                return cudf.Scalar(None, dtype=other.dtype)

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
    def __cuda_array_interface__(self) -> Mapping[str, Any]:
        output = {
            "shape": (len(self),),
            "strides": (self.dtype.itemsize,),
            "typestr": self.dtype.str,
            "data": (self.data_ptr, False),
            "version": 1,
        }

        if self.nullable and self.has_nulls():
            # Create a simple Python object that exposes the
            # `__cuda_array_interface__` attribute here since we need to modify
            # some of the attributes from the numba device array
            output["mask"] = cuda_array_interface_wrapper(
                ptr=self.mask_ptr,
                size=len(self),
                owner=self.mask,
                readonly=True,
                typestr="<t1",
            )
        return output

    def as_datetime_column(self, dtype: Dtype, **kwargs) -> DatetimeColumn:
        dtype = cudf.dtype(dtype)
        if dtype == self.dtype:
            return self
        return libcudf.unary.cast(self, dtype=dtype)

    def as_timedelta_column(
        self, dtype: Dtype, **kwargs
    ) -> "cudf.core.column.TimeDeltaColumn":
        raise TypeError(
            f"cannot astype a datetimelike from {self.dtype} to {dtype}"
        )

    def as_numerical_column(
        self, dtype: Dtype, **kwargs
    ) -> "cudf.core.column.NumericalColumn":
        return cast(
            "cudf.core.column.NumericalColumn", self.as_numerical.astype(dtype)
        )

    def as_string_column(
        self, dtype: Dtype, format=None, **kwargs
    ) -> "cudf.core.column.StringColumn":
        if format is None:
            format = _dtype_to_format_conversion.get(
                self.dtype.name, "%Y-%m-%d %H:%M:%S"
            )
        if format in _DATETIME_SPECIAL_FORMATS:
            names = as_column(_DATETIME_NAMES)
        else:
            names = cudf.core.column.column_empty(
                0, dtype="object", masked=False
            )
        if len(self) > 0:
            return string._datetime_to_str_typecast_functions[
                cudf.dtype(self.dtype)
            ](self, format, names)
        else:
            return cast(
                "cudf.core.column.StringColumn",
                column.column_empty(0, dtype="object", masked=False),
            )

    def mean(
        self, skipna=None, min_count: int = 0, dtype=np.float64
    ) -> ScalarLike:
        return pd.Timestamp(
            self.as_numerical.mean(
                skipna=skipna, min_count=min_count, dtype=dtype
            ),
            unit=self.time_unit,
        )

    def std(
        self,
        skipna: bool = None,
        min_count: int = 0,
        dtype: Dtype = np.float64,
        ddof: int = 1,
    ) -> pd.Timedelta:
        return pd.Timedelta(
            self.as_numerical.std(
                skipna=skipna, min_count=min_count, dtype=dtype, ddof=ddof
            )
            * _unit_to_nanoseconds_conversion[self.time_unit],
        )

    def median(self, skipna: bool = None) -> pd.Timestamp:
        return pd.Timestamp(
            self.as_numerical.median(skipna=skipna), unit=self.time_unit
        )

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
            return pd.Timestamp(result, unit=self.time_unit)
        return result.astype(self.dtype)

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        reflect, op = self._check_reflected_op(op)
        other = self._wrap_binop_normalization(other)
        if other is NotImplemented:
            return NotImplemented
        if isinstance(other, cudf.DateOffset):
            return other._datetime_binop(self, op, reflect=reflect)

        # We check this on `other` before reflection since we already know the
        # dtype of `self`.
        other_is_timedelta = is_timedelta64_dtype(other.dtype)
        other_is_datetime64 = not other_is_timedelta and is_datetime64_dtype(
            other.dtype
        )
        lhs, rhs = (other, self) if reflect else (self, other)
        out_dtype = None
        if op in {
            "__eq__",
            "NULL_EQUALS",
        }:
            out_dtype = cudf.dtype(np.bool_)
        elif (
            op
            in {
                "__ne__",
                "__lt__",
                "__gt__",
                "__le__",
                "__ge__",
            }
            and other_is_datetime64
        ):
            out_dtype = cudf.dtype(np.bool_)
        elif op == "__add__" and other_is_timedelta:
            # The only thing we can add to a datetime is a timedelta. This
            # operation is symmetric, i.e. we allow `datetime + timedelta` or
            # `timedelta + datetime`. Both result in DatetimeColumns.
            out_dtype = _resolve_mixed_dtypes(lhs, rhs, "datetime64")
        elif op == "__sub__":
            # Subtracting a datetime from a datetime results in a timedelta.
            if other_is_datetime64:
                out_dtype = _resolve_mixed_dtypes(lhs, rhs, "timedelta64")
            # We can subtract a timedelta from a datetime, but not vice versa.
            # Not only is subtraction antisymmetric (as is normal), it is only
            # well-defined if this operation was not invoked via reflection.
            elif other_is_timedelta and not reflect:
                out_dtype = _resolve_mixed_dtypes(lhs, rhs, "datetime64")

        if out_dtype is None:
            return NotImplemented

        return libcudf.binaryop.binaryop(lhs, rhs, op, out_dtype)

    def fillna(
        self, fill_value: Any = None, method: str = None, dtype: Dtype = None
    ) -> DatetimeColumn:
        if fill_value is not None:
            if cudf.utils.utils._isnat(fill_value):
                return _fillna_natwise(self)
            if is_scalar(fill_value):
                if not isinstance(fill_value, cudf.Scalar):
                    fill_value = cudf.Scalar(fill_value, dtype=self.dtype)
            else:
                fill_value = column.as_column(fill_value, nan_as_null=False)

        return super().fillna(fill_value, method)

    def find_first_value(
        self, value: ScalarLike, closest: bool = False
    ) -> int:
        """
        Returns offset of first value that matches
        """
        value = pd.to_datetime(value)
        value = column.as_column(
            value, dtype=self.dtype
        ).as_numerical.element_indexing(0)
        return self.as_numerical.find_first_value(value, closest=closest)

    def find_last_value(self, value: ScalarLike, closest: bool = False) -> int:
        """
        Returns offset of last value that matches
        """
        value = pd.to_datetime(value)
        value = column.as_column(
            value, dtype=self.dtype
        ).as_numerical.element_indexing(0)
        return self.as_numerical.find_last_value(value, closest=closest)

    @property
    def is_unique(self) -> bool:
        return self.as_numerical.is_unique

    def isin(self, values: Sequence) -> ColumnBase:
        return cudf.core.tools.datetimes._isin_datetimelike(self, values)

    def can_cast_safely(self, to_dtype: Dtype) -> bool:
        if np.issubdtype(to_dtype, np.datetime64):

            to_res, _ = np.datetime_data(to_dtype)
            self_res, _ = np.datetime_data(self.dtype)

            max_int = np.iinfo(cudf.dtype("int64")).max

            max_dist = np.timedelta64(
                self.max().astype(cudf.dtype("int64"), copy=False), self_res
            )
            min_dist = np.timedelta64(
                self.min().astype(cudf.dtype("int64"), copy=False), self_res
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


def infer_format(element: str, **kwargs) -> str:
    """
    Infers datetime format from a string, also takes cares for `ms` and `ns`
    """
    fmt = _guess_datetime_format(element, **kwargs)

    if fmt is not None:
        return fmt

    element_parts = element.split(".")
    if len(element_parts) != 2:
        raise ValueError("Given date string not likely a datetime.")

    # There is possibility that the element is of following format
    # '00:00:03.333333 2016-01-01'
    second_parts = re.split(r"(\D+)", element_parts[1], maxsplit=1)
    subsecond_fmt = ".%" + str(len(second_parts[0])) + "f"

    first_part = _guess_datetime_format(element_parts[0], **kwargs)
    # For the case where first_part is '00:00:03'
    if first_part is None:
        tmp = "1970-01-01 " + element_parts[0]
        first_part = _guess_datetime_format(tmp, **kwargs).split(" ", 1)[1]
    if first_part is None:
        raise ValueError("Unable to infer the timestamp format from the data")

    if len(second_parts) > 1:
        # "Z" indicates Zulu time(widely used in aviation) - Which is
        # UTC timezone that currently cudf only supports. Having any other
        # unsupported timezone will let the code fail below
        # with a ValueError.
        second_parts.remove("Z")
        second_part = "".join(second_parts[1:])

        if len(second_part) > 1:
            # Only infer if second_parts is not an empty string.
            second_part = _guess_datetime_format(second_part, **kwargs)
    else:
        second_part = ""

    try:
        fmt = first_part + subsecond_fmt + second_part
    except Exception:
        raise ValueError("Unable to infer the timestamp format from the data")

    return fmt


def _resolve_mixed_dtypes(
    lhs: ColumnBinaryOperand, rhs: ColumnBinaryOperand, base_type: str
) -> Dtype:
    units = ["s", "ms", "us", "ns"]
    lhs_time_unit = cudf.utils.dtypes.get_time_unit(lhs)
    lhs_unit = units.index(lhs_time_unit)
    rhs_time_unit = cudf.utils.dtypes.get_time_unit(rhs)
    rhs_unit = units.index(rhs_time_unit)
    return cudf.dtype(f"{base_type}[{units[max(lhs_unit, rhs_unit)]}]")
