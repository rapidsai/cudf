# Copyright (c) 2019-2021, NVIDIA CORPORATION.

from __future__ import annotations

import builtins
import datetime as dt
import re
from numbers import Number
from types import SimpleNamespace
from typing import Any, Mapping, Sequence, Union, cast

import numpy as np
import pandas as pd

import cudf
from cudf import _lib as libcudf
from cudf._typing import DatetimeLikeScalar, Dtype, DtypeObj, ScalarLike
from cudf.core._compat import PANDAS_GE_120
from cudf.core.buffer import Buffer
from cudf.core.column import (
    ColumnBase,
    as_column,
    column,
    column_empty_like,
    string,
)
from cudf.utils.dtypes import is_scalar
from cudf.utils.utils import _fillna_natwise

if PANDAS_GE_120:
    _guess_datetime_format = pd.core.tools.datetimes.guess_datetime_format
else:
    _guess_datetime_format = pd.core.tools.datetimes._guess_datetime_format

# nanoseconds per time_unit
_numpy_to_pandas_conversion = {
    "ns": 1,
    "us": 1000,
    "ms": 1000000,
    "s": 1000000000,
    "m": 60000000000,
    "h": 3600000000000,
    "D": 86400000000000,
}

_dtype_to_format_conversion = {
    "datetime64[ns]": "%Y-%m-%d %H:%M:%S.%9f",
    "datetime64[us]": "%Y-%m-%d %H:%M:%S.%6f",
    "datetime64[ms]": "%Y-%m-%d %H:%M:%S.%3f",
    "datetime64[s]": "%Y-%m-%d %H:%M:%S",
}


class DatetimeColumn(column.ColumnBase):
    def __init__(
        self,
        data: Buffer,
        dtype: DtypeObj,
        mask: Buffer = None,
        size: int = None,  # TODO: make non-optional
        offset: int = 0,
        null_count: int = None,
    ):
        """
        Parameters
        ----------
        data : Buffer
            The datetime values
        dtype : np.dtype
            The data type
        mask : Buffer; optional
            The validity mask
        """
        dtype = np.dtype(dtype)
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
            self.astype("datetime64[ns]").to_array("NAT"),
            copy=False,
            index=index,
        )

    def get_dt_field(self, field: str) -> ColumnBase:
        return libcudf.datetime.extract_datetime_component(self, field)

    def normalize_binop_value(self, other: DatetimeLikeScalar) -> ScalarLike:
        if isinstance(other, cudf.Scalar):
            return other

        if isinstance(other, np.ndarray) and other.ndim == 0:
            other = other.item()

        if isinstance(other, dt.datetime):
            other = np.datetime64(other)
        elif isinstance(other, dt.timedelta):
            other = np.timedelta64(other)
        elif isinstance(other, pd.Timestamp):
            other = other.to_datetime64()
        elif isinstance(other, pd.Timedelta):
            other = other.to_timedelta64()
        elif isinstance(other, cudf.DateOffset):
            return other
        if isinstance(other, np.datetime64):
            if np.isnat(other):
                return cudf.Scalar(None, dtype=self.dtype)

            other = other.astype(self.dtype)
            return cudf.Scalar(other)
        elif isinstance(other, np.timedelta64):
            other_time_unit = cudf.utils.dtypes.get_time_unit(other)

            if other_time_unit not in ("s", "ms", "ns", "us"):
                other = other.astype("timedelta64[s]")

            if np.isnat(other):
                return cudf.Scalar(None, dtype=other.dtype)

            return cudf.Scalar(other)
        elif other is None:
            return cudf.Scalar(other, dtype=self.dtype)
        else:
            raise TypeError(f"cannot normalize {type(other)}")

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
    def __cuda_array_interface__(self) -> Mapping[builtins.str, Any]:
        output = {
            "shape": (len(self),),
            "strides": (self.dtype.itemsize,),
            "typestr": self.dtype.str,
            "data": (self.data_ptr, False),
            "version": 1,
        }

        if self.nullable and self.has_nulls:

            # Create a simple Python object that exposes the
            # `__cuda_array_interface__` attribute here since we need to modify
            # some of the attributes from the numba device array
            mask = SimpleNamespace(
                __cuda_array_interface__={
                    "shape": (len(self),),
                    "typestr": "<t1",
                    "data": (self.mask_ptr, True),
                    "version": 1,
                }
            )
            output["mask"] = mask

        return output

    def as_datetime_column(self, dtype: Dtype, **kwargs) -> DatetimeColumn:
        dtype = np.dtype(dtype)
        if dtype == self.dtype:
            return self
        return libcudf.unary.cast(self, dtype=dtype)

    def as_timedelta_column(
        self, dtype: Dtype, **kwargs
    ) -> "cudf.core.column.TimeDeltaColumn":
        raise TypeError(
            f"cannot astype a datetimelike from [{self.dtype}] to [{dtype}]"
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
        if len(self) > 0:
            return string._datetime_to_str_typecast_functions[
                np.dtype(self.dtype)
            ](self, format)
        else:
            return cast(
                "cudf.core.column.StringColumn",
                column.column_empty(0, dtype="object", masked=False),
            )

    def default_na_value(self) -> DatetimeLikeScalar:
        """Returns the default NA value for this column
        """
        return np.datetime64("nat", self.time_unit)

    def mean(self, skipna=None, dtype=np.float64) -> ScalarLike:
        return pd.Timestamp(
            self.as_numerical.mean(skipna=skipna, dtype=dtype),
            unit=self.time_unit,
        )

    def std(
        self, skipna: bool = None, ddof: int = 1, dtype: Dtype = np.float64
    ) -> pd.Timedelta:
        return pd.Timedelta(
            self.as_numerical.std(skipna=skipna, ddof=ddof, dtype=dtype)
            * _numpy_to_pandas_conversion[self.time_unit],
        )

    def median(self, skipna: bool = None) -> pd.Timestamp:
        return pd.Timestamp(
            self.as_numerical.median(skipna=skipna), unit=self.time_unit
        )

    def quantile(
        self, q: Union[float, Sequence[float]], interpolation: str, exact: bool
    ) -> ColumnBase:
        result = self.as_numerical.quantile(
            q=q, interpolation=interpolation, exact=exact
        )
        if isinstance(q, Number):
            return pd.Timestamp(result, unit=self.time_unit)
        return result.astype(self.dtype)

    def binary_operator(
        self,
        op: str,
        rhs: Union[ColumnBase, "cudf.Scalar"],
        reflect: bool = False,
    ) -> ColumnBase:
        if isinstance(rhs, cudf.DateOffset):
            return rhs._datetime_binop(self, op, reflect=reflect)
        lhs: Union[ScalarLike, ColumnBase] = self
        if op in ("eq", "ne", "lt", "gt", "le", "ge", "NULL_EQUALS"):
            out_dtype = np.dtype(np.bool_)  # type: Dtype
        elif op == "add" and pd.api.types.is_timedelta64_dtype(rhs.dtype):
            out_dtype = cudf.core.column.timedelta._timedelta_add_result_dtype(
                rhs, lhs
            )
        elif op == "sub" and pd.api.types.is_timedelta64_dtype(rhs.dtype):
            out_dtype = cudf.core.column.timedelta._timedelta_sub_result_dtype(
                rhs if reflect else lhs, lhs if reflect else rhs
            )
        elif op == "sub" and pd.api.types.is_datetime64_dtype(rhs.dtype):
            units = ["s", "ms", "us", "ns"]
            lhs_time_unit = cudf.utils.dtypes.get_time_unit(lhs)
            lhs_unit = units.index(lhs_time_unit)
            rhs_time_unit = cudf.utils.dtypes.get_time_unit(rhs)
            rhs_unit = units.index(rhs_time_unit)
            out_dtype = np.dtype(
                f"timedelta64[{units[max(lhs_unit, rhs_unit)]}]"
            )
        else:
            raise TypeError(
                f"Series of dtype {self.dtype} cannot perform "
                f" the operation {op}"
            )

        if reflect:
            lhs, rhs = rhs, lhs
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
        value = column.as_column(value, dtype=self.dtype).as_numerical[0]
        return self.as_numerical.find_first_value(value, closest=closest)

    def find_last_value(self, value: ScalarLike, closest: bool = False) -> int:
        """
        Returns offset of last value that matches
        """
        value = pd.to_datetime(value)
        value = column.as_column(value, dtype=self.dtype).as_numerical[0]
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

            max_int = np.iinfo(np.dtype("int64")).max

            max_dist = np.timedelta64(
                self.max().astype(np.dtype("int64"), copy=False), self_res
            )
            min_dist = np.timedelta64(
                self.min().astype(np.dtype("int64"), copy=False), self_res
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
        elif to_dtype == np.dtype("int64") or to_dtype == np.dtype("O"):
            # can safely cast to representation, or string
            return True
        else:
            return False

    def _make_copy_with_na_as_null(self):
        """Return a copy with NaN values replaced with nulls."""
        null = column_empty_like(self, masked=True, newsize=1)
        out_col = cudf._lib.replace.replace(
            self,
            as_column(
                Buffer(
                    np.array([self.default_na_value()], dtype=self.dtype).view(
                        "|u1"
                    )
                ),
                dtype=self.dtype,
            ),
            null,
        )
        return out_col


def binop_offset(lhs, rhs, op):
    if rhs._is_no_op:
        return lhs
    else:
        rhs = rhs._generate_column(len(lhs), op)
        out = libcudf.datetime.add_months(lhs, rhs)
        return out


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
        # unsuppported timezone will let the code fail below
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
