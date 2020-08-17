# Copyright (c) 2019-2020, NVIDIA CORPORATION.
import datetime as dt
import re

import numpy as np
import pandas as pd
import pyarrow as pa

import cudf
from cudf import _lib as libcudf
from cudf._lib.nvtx import annotate
from cudf._lib.scalar import Scalar, as_scalar
from cudf.core.column import column, string
from cudf.utils.dtypes import is_scalar, np_to_pa_dtype
from cudf.utils.utils import buffers_from_pyarrow

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
        self, data, dtype, mask=None, size=None, offset=0, null_count=None
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

    def __contains__(self, item):
        try:
            item = np.datetime64(item, self._time_unit)
        except ValueError:
            # If item cannot be converted to datetime type
            # np.datetime64 raises ValueError, hence `item`
            # cannot exist in `self`.
            return False
        return item.astype("int64") in self.as_numerical

    @property
    def time_unit(self):
        return self._time_unit

    @property
    def year(self):
        return self.get_dt_field("year")

    @property
    def month(self):
        return self.get_dt_field("month")

    @property
    def day(self):
        return self.get_dt_field("day")

    @property
    def hour(self):
        return self.get_dt_field("hour")

    @property
    def minute(self):
        return self.get_dt_field("minute")

    @property
    def second(self):
        return self.get_dt_field("second")

    @property
    def weekday(self):
        return self.get_dt_field("weekday")

    def get_dt_field(self, field):
        return libcudf.datetime.extract_datetime_component(self, field)

    def normalize_binop_value(self, other):
        if isinstance(other, dt.datetime):
            other = np.datetime64(other)
        elif isinstance(other, dt.timedelta):
            other = np.timedelta64(other)
        elif isinstance(other, pd.Timestamp):
            other = other.to_datetime64()
        elif isinstance(other, pd.Timedelta):
            other = other.to_timedelta64()

        if isinstance(other, np.datetime64):
            if np.isnat(other):
                return as_scalar(val=None, dtype=self.dtype)

            other = other.astype(self.dtype)
            return as_scalar(other)
        elif isinstance(other, np.timedelta64):
            other_time_unit = cudf.utils.dtypes.get_time_unit(other)

            if other_time_unit not in ("s", "ms", "ns", "us"):
                other = other.astype("timedelta64[s]")

            if np.isnat(other):
                return as_scalar(val=None, dtype=other.dtype)

            return as_scalar(other)
        else:
            raise TypeError("cannot normalize {}".format(type(other)))

    @property
    def as_numerical(self):
        return column.build_column(
            data=self.base_data,
            dtype=np.int64,
            mask=self.base_mask,
            offset=self.offset,
            size=self.size,
        )

    def as_datetime_column(self, dtype, **kwargs):
        dtype = np.dtype(dtype)
        if dtype == self.dtype:
            return self
        return libcudf.unary.cast(self, dtype=dtype)

    def as_timedelta_column(self, dtype, **kwargs):
        raise TypeError(
            f"cannot astype a datetimelike from [{self.dtype}] to [{dtype}]"
        )

    def as_numerical_column(self, dtype, **kwargs):
        return self.as_numerical.astype(dtype)

    def as_string_column(self, dtype, **kwargs):

        if not kwargs.get("format"):
            fmt = _dtype_to_format_conversion.get(
                self.dtype.name, "%Y-%m-%d %H:%M:%S"
            )
            kwargs["format"] = fmt
        if len(self) > 0:
            return string._numeric_to_str_typecast_functions[
                np.dtype(self.dtype)
            ](self, **kwargs)
        else:
            return column.column_empty(0, dtype="object", masked=False)

    def to_pandas(self, index=None, nullable_pd_dtype=False):
        if nullable_pd_dtype:
            raise NotImplementedError(
                f"nullable_pd_dtype=True is not supported for {self.dtype}"
            )

        return pd.Series(
            self.to_array(fillna="pandas").astype(self.dtype), index=index
        )

    def to_arrow(self):
        mask = None
        if self.nullable:
            mask = pa.py_buffer(self.mask_array_view.copy_to_host())
        data = pa.py_buffer(self.as_numerical.data_array_view.copy_to_host())
        pa_dtype = np_to_pa_dtype(self.dtype)
        return pa.Array.from_buffers(
            type=pa_dtype,
            length=len(self),
            buffers=[mask, data],
            null_count=self.null_count,
        )

    def default_na_value(self):
        """Returns the default NA value for this column
        """
        return np.datetime64("nat", self.time_unit)

    def binary_operator(self, op, rhs, reflect=False):
        lhs, rhs = self, rhs
        if op in ("eq", "ne", "lt", "gt", "le", "ge"):
            out_dtype = np.bool
        elif op == "add" and pd.api.types.is_timedelta64_dtype(rhs.dtype):
            out_dtype = cudf.core.column.timedelta._timedelta_binary_op_add(
                rhs, lhs
            )
        elif op == "sub" and pd.api.types.is_timedelta64_dtype(rhs.dtype):
            out_dtype = cudf.core.column.timedelta._timedelta_binary_op_sub(
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

        return binop(lhs, rhs, op=op, out_dtype=out_dtype)

    def fillna(self, fill_value):
        if is_scalar(fill_value):
            if not isinstance(fill_value, Scalar):
                fill_value = np.datetime64(fill_value, self.time_unit)
        else:
            fill_value = column.as_column(fill_value, nan_as_null=False)

        result = libcudf.replace.replace_nulls(self, fill_value)
        if isinstance(fill_value, np.datetime64) and np.isnat(fill_value):
            # If the value we are filling is np.datetime64("NAT")
            # we set the same mask as current column.
            # However where there are "<NA>" in the
            # columns, their corresponding locations
            # in base_data will contain min(int64) values.

            return column.build_column(
                data=result.base_data,
                dtype=result.dtype,
                mask=self.base_mask,
                size=result.size,
                offset=result.offset,
                children=result.base_children,
            )
        return result

    def find_first_value(self, value, closest=False):
        """
        Returns offset of first value that matches
        """
        value = pd.to_datetime(value)
        value = column.as_column(value).as_numerical[0]
        return self.as_numerical.find_first_value(value, closest=closest)

    def find_last_value(self, value, closest=False):
        """
        Returns offset of last value that matches
        """
        value = pd.to_datetime(value)
        value = column.as_column(value).as_numerical[0]
        return self.as_numerical.find_last_value(value, closest=closest)

    @property
    def is_unique(self):
        return self.as_numerical.is_unique

    @classmethod
    def from_arrow(cls, array, dtype=None):
        if dtype is None:
            dtype = np.dtype("M8[{}]".format(array.type.unit))

        pa_size, pa_offset, pamask, padata, _ = buffers_from_pyarrow(array)

        return DatetimeColumn(
            data=padata,
            mask=pamask,
            dtype=dtype,
            size=pa_size,
            offset=pa_offset,
        )

    def can_cast_safely(self, to_dtype):
        if np.issubdtype(to_dtype, np.datetime64):

            to_res, _ = np.datetime_data(to_dtype)
            self_res, _ = np.datetime_data(self.dtype)

            max_int = np.iinfo(np.dtype("int64")).max

            max_dist = self.max().astype(np.timedelta64, copy=False)
            min_dist = self.min().astype(np.timedelta64, copy=False)

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


@annotate("BINARY_OP", color="orange", domain="cudf_python")
def binop(lhs, rhs, op, out_dtype):
    out = libcudf.binaryop.binaryop(lhs, rhs, op, out_dtype)
    return out


def infer_format(element, **kwargs):
    """
    Infers datetime format from a string, also takes cares for `ms` and `ns`
    """

    fmt = pd.core.tools.datetimes._guess_datetime_format(element, **kwargs)

    if fmt is not None:
        return fmt

    element_parts = element.split(".")
    if len(element_parts) != 2:
        raise ValueError("Given date string not likely a datetime.")

    # There is possibility that the element is of following format
    # '00:00:03.333333 2016-01-01'
    second_part = re.split(r"(\D+)", element_parts[1], maxsplit=1)
    subsecond_fmt = ".%" + str(len(second_part[0])) + "f"

    first_part = pd.core.tools.datetimes._guess_datetime_format(
        element_parts[0], **kwargs
    )
    # For the case where first_part is '00:00:03'
    if first_part is None:
        tmp = "1970-01-01 " + element_parts[0]
        first_part = pd.core.tools.datetimes._guess_datetime_format(
            tmp, **kwargs
        ).split(" ", 1)[1]
    if first_part is None:
        raise ValueError("Unable to infer the timestamp format from the data")

    if len(second_part) > 1:
        second_part = pd.core.tools.datetimes._guess_datetime_format(
            "".join(second_part[1:]), **kwargs
        )
    else:
        second_part = ""

    try:
        fmt = first_part + subsecond_fmt + second_part
    except Exception:
        raise ValueError("Unable to infer the timestamp format from the data")

    return fmt
