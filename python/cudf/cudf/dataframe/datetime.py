import datetime as dt

import numpy as np
import pandas as pd
import pyarrow as pa

import cudf.bindings.binops as cpp_binops
import cudf.bindings.copying as cpp_copying
import cudf.bindings.reduce as cpp_reduce
import cudf.bindings.replace as cpp_replace
import cudf.bindings.unaryops as cpp_unaryops
from cudf._sort import get_sorted_inds
from cudf.bindings.cudf_cpp import get_ctype_ptr, np_to_pa_dtype
from cudf.bindings.nvtx import nvtx_range_pop, nvtx_range_push
from cudf.comm.serialize import register_distributed_serializer
from cudf.dataframe import columnops
from cudf.dataframe.buffer import Buffer
from cudf.dataframe.numerical import NumericalColumn
from cudf.utils import cudautils, utils
from cudf.utils.utils import is_single_value


class DatetimeColumn(columnops.TypedColumnBase):
    # TODO - we only support milliseconds (date64)
    # we should support date32 and timestamp, but perhaps
    # only after we move to arrow
    # we also need to support other formats besides Date64
    _npdatetime64_dtype = np.dtype("datetime64[ms]")

    def __init__(
        self, data, mask=None, null_count=None, dtype=None, name=None
    ):
        super(DatetimeColumn, self).__init__(
            data=data, mask=mask, null_count=null_count, dtype=dtype, name=name
        )
        self._precision = 1e-3
        self._inverse_precision = 1e3
        self._pandas_conversion_factor = 1e9 * self._precision

    def serialize(self, serialize):
        header, frames = super(DatetimeColumn, self).serialize(serialize)
        assert "dtype" not in header
        header["dtype"] = serialize(self._dtype)
        return header, frames

    @classmethod
    def deserialize(cls, deserialize, header, frames):
        data, mask = super(DatetimeColumn, cls).deserialize(
            deserialize, header, frames
        )
        col = cls(
            data=data,
            mask=mask,
            null_count=header["null_count"],
            dtype=deserialize(*header["dtype"]),
        )
        return col

    @classmethod
    def from_numpy(cls, array):
        array = array.astype(cls._npdatetime64_dtype)
        assert array.dtype.itemsize == 8
        buf = Buffer(array)
        return cls(data=buf, dtype=buf.dtype)

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
            ary = utils.scalar_broadcast_to(
                other.value * self._pandas_conversion_factor,
                shape=len(self),
                dtype=self._npdatetime64_dtype,
            )
        elif isinstance(other, np.datetime64):
            other = other.astype(self._npdatetime64_dtype)
            ary = utils.scalar_broadcast_to(
                other, shape=len(self), dtype=self._npdatetime64_dtype
            )
        else:
            raise TypeError("cannot broadcast {}".format(type(other)))

        buf = Buffer(ary)
        result = self.replace(data=buf, dtype=self.dtype)
        return result

    @property
    def as_numerical(self):
        from cudf.dataframe import numerical

        return self.view(
            numerical.NumericalColumn,
            dtype="int64",
            data=self.data.astype("int64"),
        )

    def astype(self, dtype):
        from cudf.dataframe import string

        if self.dtype is dtype:
            return self
        elif dtype == np.dtype("object") or np.issubdtype(
            dtype, np.dtype("U").type
        ):
            if len(self) > 0:
                dev_array = self.data.mem
                dev_ptr = get_ctype_ptr(dev_array)
                null_ptr = None
                if self.mask is not None:
                    null_ptr = get_ctype_ptr(self.mask.mem)
                kwargs = {
                    "count": len(self),
                    "nulls": null_ptr,
                    "bdevmem": True,
                    "units": "ms",
                }
                data = string._numeric_to_str_typecast_functions[
                    np.dtype(self.dtype)
                ](dev_ptr, **kwargs)

            else:
                data = []

            return string.StringColumn(data=data)

        return self.as_numerical.astype(dtype)

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
        data = pa.py_buffer(self.data.mem.copy_to_host().view("int64"))
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
            return np.datetime64("nat", "ms")
        else:
            raise TypeError(
                "datetime column of {} has no NaN value".format(self.dtype)
            )

    def fillna(self, fill_value, inplace=False):
        if is_single_value(fill_value):
            fill_value = np.datetime64(fill_value, "ms")
        else:
            fill_value = columnops.as_column(fill_value, nan_as_null=False)

        result = cpp_replace.apply_replace_nulls(self, fill_value)

        result = result.replace(mask=None)
        return self._mimic_inplace(result, inplace)

    def sort_by_values(self, ascending=True, na_position="last"):
        sort_inds = get_sorted_inds(self, ascending, na_position)
        col_keys = cpp_copying.apply_gather_column(self, sort_inds.data.mem)
        col_inds = self.replace(
            data=sort_inds.data,
            mask=sort_inds.mask,
            dtype=sort_inds.data.dtype,
        )
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

    def unique(self, method="sort"):
        # method variable will indicate what algorithm to use to
        # calculate unique, not used right now
        if method != "sort":
            msg = "non sort based unique() not implemented yet"
            raise NotImplementedError(msg)
        segs, sortedvals = self._unique_segments()
        # gather result
        out_col = cpp_copying.apply_gather_array(sortedvals, segs)
        return out_col

    def value_counts(self, method="sort"):
        if method != "sort":
            msg = "non sort based value_count() not implemented yet"
            raise NotImplementedError(msg)
        segs, sortedvals = self._unique_segments()
        # Return both values and their counts
        out_vals = cpp_copying.apply_gather_array(sortedvals, segs)
        out2 = cudautils.value_count(segs, len(sortedvals))
        out_counts = NumericalColumn(data=Buffer(out2), dtype=np.intp)
        return out_vals, out_counts

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


register_distributed_serializer(DatetimeColumn)
