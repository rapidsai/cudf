import datetime as dt
import pickle

import numpy as np
import pandas as pd
import pyarrow as pa

import cudf.bindings.binops as cpp_binops
import cudf.bindings.copying as cpp_copying
import cudf.bindings.reduce as cpp_reduce
import cudf.bindings.replace as cpp_replace
import cudf.bindings.search as cpp_search
import cudf.bindings.unaryops as cpp_unaryops
from cudf._sort import get_sorted_inds
from cudf.bindings.cudf_cpp import get_ctype_ptr, np_to_pa_dtype
from cudf.bindings.nvtx import nvtx_range_pop, nvtx_range_push
from cudf.dataframe import columnops
from cudf.dataframe.buffer import Buffer
from cudf.utils import utils
from cudf.utils.utils import is_scalar

# nanoseconds per time_unit
_numpy_to_pandas_conversion = {
    "ns": 1,
    "us": 1000,
    "ms": 1000000,
    "s": 1000000000,
    "D": 1000000000 * 86400,
}


class DatetimeColumn(columnops.TypedColumnBase):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        data : Buffer
            The datetime values
        mask : Buffer; optional
            The validity mask
        null_count : int; optional
            The number of null values in the mask.
        dtype : np.dtype
            Data type
        name : str
            The Column name
        """
        super(DatetimeColumn, self).__init__(**kwargs)
        assert self.dtype.type is np.datetime64
        self._time_unit, _ = np.datetime_data(self.dtype)

    def serialize(self):
        header, frames = super(DatetimeColumn, self).serialize()
        header["type"] = pickle.dumps(type(self))
        header["dtype"] = self._dtype.str
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        data, mask = super(DatetimeColumn, cls).deserialize(header, frames)
        dtype = header["dtype"]
        col = cls(
            data=data, mask=mask, null_count=header["null_count"], dtype=dtype
        )
        return col

    @classmethod
    def from_numpy(cls, array):
        cast_dtype = array.dtype.type == np.int64
        if array.dtype.kind == "M":
            time_unit, _ = np.datetime_data(array.dtype)
            cast_dtype = time_unit == "D" or (
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
            array = array.astype(np.dtype("datetime64[ms]"))
        assert array.dtype.itemsize == 8
        return cls(data=Buffer(array), dtype=array.dtype)

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

    def get_dt_field(self, field):
        out = columnops.column_empty_like_same_mask(self, dtype=np.int16)
        cpp_unaryops.apply_dt_extract_op(self, out, field)
        out.name = self.name
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
                other, shape=len(self), dtype=self.dtype
            )
        else:
            raise TypeError("cannot broadcast {}".format(type(other)))

        return self.replace(data=Buffer(ary), dtype=self.dtype)

    @property
    def as_numerical(self):
        from cudf.dataframe import numerical

        data = Buffer(self.data.mem.view(np.int64))
        return self.view(
            numerical.NumericalColumn, data=data, dtype=data.dtype
        )

    def as_datetime_column(self, dtype, **kwargs):
        import cudf.bindings.typecast as typecast

        dtype = np.dtype(dtype)
        if dtype == self.dtype:
            return self
        return typecast.apply_cast(self, dtype=dtype)

    def as_numerical_column(self, dtype, **kwargs):
        return self.as_numerical.astype(dtype)

    def as_string_column(self, dtype, **kwargs):
        from cudf.dataframe import string

        if len(self) > 0:
            dev_array = self.data.mem
            dev_ptr = get_ctype_ptr(dev_array)
            null_ptr = None
            if self.mask is not None:
                null_ptr = get_ctype_ptr(self.mask.mem)
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

        else:
            data = []

        return string.StringColumn(data=data)

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
        if self.has_null_mask:
            mask = pa.py_buffer(self.nullmask.mem.copy_to_host())
        data = pa.py_buffer(self.as_numerical.data.mem.copy_to_host())
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
            fill_value = columnops.as_column(fill_value, nan_as_null=False)

        result = cpp_replace.apply_replace_nulls(self, fill_value)

        result = result.replace(mask=None)
        return self._mimic_inplace(result, inplace)

    def sort_by_values(self, ascending=True, na_position="last"):
        col_inds = get_sorted_inds(self, ascending, na_position)
        col_keys = cpp_copying.apply_gather(self, col_inds)
        col_inds.name = self.name
        return col_keys, col_inds

    def min(self, dtype=None):
        return cpp_reduce.apply_reduce("min", self, dtype=dtype)

    def max(self, dtype=None):
        return cpp_reduce.apply_reduce("max", self, dtype=dtype)

    def find_first_value(self, value):
        """
        Returns offset of first value that matches
        """
        value = pd.to_datetime(value)
        value = columnops.as_column(value).as_numerical[0]
        return self.as_numerical.find_first_value(value)

    def find_last_value(self, value):
        """
        Returns offset of last value that matches
        """
        value = pd.to_datetime(value)
        value = columnops.as_column(value).as_numerical[0]
        return self.as_numerical.find_last_value(value)

    def searchsorted(self, value, side="left"):
        value_col = columnops.as_column(value)
        return cpp_search.search_sorted(self, value_col, side)

    def unique(self, method="sort"):
        # method variable will indicate what algorithm to use to
        # calculate unique, not used right now
        if method != "sort":
            msg = "non sort based unique() not implemented yet"
            raise NotImplementedError(msg)
        segs, sortedvals = self._unique_segments()
        # gather result
        out_col = cpp_copying.apply_gather(sortedvals, segs)
        return out_col

    @property
    def is_unique(self):
        return self.as_numerical.is_unique

    @property
    def is_monotonic_increasing(self):
        if not hasattr(self, "_is_monotonic_increasing"):
            self._is_monotonic_increasing = binop(
                self[1:], self[:-1], "ge", "bool"
            ).all()
        return self._is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        if not hasattr(self, "_is_monotonic_decreasing"):
            self._is_monotonic_decreasing = binop(
                self[1:], self[:-1], "le", "bool"
            ).all()
        return self._is_monotonic_decreasing


def binop(lhs, rhs, op, out_dtype):
    nvtx_range_push("CUDF_BINARY_OP", "orange")
    masked = lhs.has_null_mask or rhs.has_null_mask
    out = columnops.column_empty_like(lhs, dtype=out_dtype, masked=masked)
    null_count = cpp_binops.apply_op(lhs, rhs, out, op)
    out = out.replace(null_count=null_count)
    nvtx_range_pop()
    return out
