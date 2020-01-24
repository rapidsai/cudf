import datetime as dt

import numpy as np
import pandas as pd
import pyarrow as pa

import cudf
import cudf._lib as libcudf
from cudf.core._sort import get_sorted_inds
from cudf.core.buffer import Buffer
from cudf.core.column import as_column, column
from cudf.utils import utils
from cudf.utils.dtypes import is_scalar, np_to_pa_dtype

# nanoseconds per time_unit
_numpy_to_pandas_conversion = {
    "ns": 1,
    "us": 1000,
    "ms": 1000000,
    "s": 1000000000,
    "D": 1000000000 * 86400,
}


class DatetimeColumn(column.ColumnBase):
    def __init__(self, data, dtype, mask=None, offset=0):
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
        size = data.size // dtype.itemsize
        super().__init__(data, size=size, dtype=dtype, mask=mask)
        assert self.dtype.type is np.datetime64
        self._time_unit, _ = np.datetime_data(self.dtype)

    def __contains__(self, item):
        # Handles improper item types
        try:
            item = np.datetime64(item, self._time_unit)
        except Exception:
            return False
        return item.astype("int_") in self.as_numerical

    @classmethod
    def from_numpy(cls, array):
        cast_dtype = array.dtype.type == np.int64
        if array.dtype.kind == "M":
            time_unit, _ = np.datetime_data(array.dtype)
            cast_dtype = time_unit in ("D", "W", "M", "Y") or (
                len(array) > 0
                and (
                    isinstance(array[0], str)
                    or isinstance(array[0], dt.datetime)
                )
            )
        elif not cast_dtype:
            raise ValueError(
                ("Cannot infer datetime dtype " + "from np.array dtype `%s`")
                % (array.dtype)
            )

        if cast_dtype:
            array = array.astype(np.dtype("datetime64[s]"))
        assert array.dtype.itemsize == 8

        mask = None
        if np.any(np.isnat(array)):
            null = cudf.core.column.column_empty_like(
                array, masked=True, newsize=1
            )
            col = libcudf.replace.replace(
                as_column(Buffer(array), dtype=array.dtype),
                as_column(
                    Buffer(
                        np.array([np.datetime64("NaT")], dtype=array.dtype)
                    ),
                    dtype=array.dtype,
                ),
                null,
            )
            mask = col.mask

        return cls(data=Buffer(array), mask=mask, dtype=array.dtype)

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
        out = column.column_empty_like_same_mask(self, dtype=np.int16)
        libcudf.unaryops.apply_dt_extract_op(self, out, field)
        return out

    def normalize_binop_value(self, other):
        if isinstance(other, dt.datetime):
            other = np.datetime64(other)

        if isinstance(other, pd.Timestamp):
            m = _numpy_to_pandas_conversion[self.time_unit]
            ary = utils.scalar_broadcast_to(
                other.value * m, shape=len(self), dtype=self.dtype
            )
        elif isinstance(other, np.datetime64):
            other = other.astype(self.dtype)
            ary = utils.scalar_broadcast_to(
                other, size=len(self), dtype=self.dtype
            )
        else:
            raise TypeError("cannot broadcast {}".format(type(other)))

        return column.build_column(data=Buffer(ary), dtype=self.dtype)

    @property
    def as_numerical(self):
        from cudf.core.column import build_column

        return build_column(data=self.data, dtype=np.int64, mask=self.mask)

    def as_datetime_column(self, dtype, **kwargs):
        dtype = np.dtype(dtype)
        if dtype == self.dtype:
            return self
        return libcudf.typecast.cast(self, dtype=dtype)

    def as_numerical_column(self, dtype, **kwargs):
        return self.as_numerical.astype(dtype)

    def as_string_column(self, dtype, **kwargs):
        from cudf.core.column import string

        if len(self) > 0:
            dev_ptr = self.data.ptr
            null_ptr = None
            if self.nullable:
                null_ptr = self.mask.ptr
            kwargs.update(
                {
                    "count": len(self),
                    "nulls": null_ptr,
                    "bdevmem": True,
                    "units": self.time_unit,
                }
            )
            data = string._numeric_to_str_typecast_functions[
                np.dtype(self.dtype)
            ](dev_ptr, **kwargs)
            return as_column(data)
        else:
            return column.column_empty(0, dtype="object", masked=False)

    def unordered_compare(self, cmpop, rhs):
        lhs, rhs = self, rhs
        return binop(lhs, rhs, op=cmpop, out_dtype=np.bool)

    def ordered_compare(self, cmpop, rhs):
        lhs, rhs = self, rhs
        return binop(lhs, rhs, op=cmpop, out_dtype=np.bool)

    def to_pandas(self, index=None):
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
        dkind = self.dtype.kind
        if dkind == "M":
            return np.datetime64("nat", self.time_unit)
        else:
            raise TypeError(
                "datetime column of {} has no NaN value".format(self.dtype)
            )

    def fillna(self, fill_value, inplace=False):
        if is_scalar(fill_value):
            fill_value = np.datetime64(fill_value, self.time_unit)
        else:
            fill_value = column.as_column(fill_value, nan_as_null=False)

        result = libcudf.replace.replace_nulls(self, fill_value)

        result.mask = None
        return self._mimic_inplace(result, inplace)

    def sort_by_values(self, ascending=True, na_position="last"):
        col_inds = get_sorted_inds(self, ascending, na_position)
        col_keys = self[col_inds]
        return col_keys, col_inds

    def min(self, dtype=None):
        return libcudf.reduce.reduce("min", self, dtype=dtype)

    def max(self, dtype=None):
        return libcudf.reduce.reduce("max", self, dtype=dtype)

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

    def searchsorted(self, value, side="left"):
        value_col = column.as_column(value)
        return libcudf.search.search_sorted(self, value_col, side)

    def unique(self, method="sort"):
        # method variable will indicate what algorithm to use to
        # calculate unique, not used right now
        if method != "sort":
            msg = "non sort based unique() not implemented yet"
            raise NotImplementedError(msg)
        segs, sortedvals = self._unique_segments()
        # gather result
        out_col = column.as_column(sortedvals)[segs]
        return out_col

    @property
    def is_unique(self):
        return self.as_numerical.is_unique

    @property
    def is_monotonic_increasing(self):
        if not hasattr(self, "_is_monotonic_increasing"):
            if self.nullable and self.has_nulls:
                self._is_monotonic_increasing = False
            else:
                self._is_monotonic_increasing = libcudf.issorted.issorted(
                    [self]
                )
        return self._is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        if not hasattr(self, "_is_monotonic_decreasing"):
            if self.nullable and self.has_nulls:
                self._is_monotonic_decreasing = False
            else:
                self._is_monotonic_decreasing = libcudf.issorted.issorted(
                    [self], [1]
                )
        return self._is_monotonic_decreasing


def binop(lhs, rhs, op, out_dtype):
    libcudf.nvtx.nvtx_range_push("CUDF_BINARY_OP", "orange")
    masked = lhs.nullable or rhs.nullable
    out = column.column_empty_like(lhs, dtype=out_dtype, masked=masked)
    _ = libcudf.binops.apply_op(lhs, rhs, out, op)
    libcudf.nvtx.nvtx_range_pop()
    return out
