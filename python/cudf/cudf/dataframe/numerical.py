# Copyright (c) 2018-2019, NVIDIA CORPORATION.

from __future__ import division, print_function

import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.types import is_integer_dtype

from librmm_cffi import librmm as rmm

import cudf.bindings.binops as cpp_binops
import cudf.bindings.copying as cpp_copying
import cudf.bindings.hash as cpp_hash
import cudf.bindings.reduce as cpp_reduce
import cudf.bindings.replace as cpp_replace
import cudf.bindings.search as cpp_search
import cudf.bindings.sort as cpp_sort
import cudf.bindings.unaryops as cpp_unaryops
from cudf._sort import get_sorted_inds
from cudf.bindings.cudf_cpp import get_ctype_ptr, np_to_pa_dtype
from cudf.bindings.nvtx import nvtx_range_pop, nvtx_range_push
from cudf.dataframe import columnops
from cudf.dataframe.buffer import Buffer
from cudf.utils import cudautils, utils


class NumericalColumn(columnops.TypedColumnBase):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        data : Buffer
            The code values
        mask : Buffer; optional
            The validity mask
        null_count : int; optional
            The number of null values in the mask.
        dtype : np.dtype
            Data type
        """
        super(NumericalColumn, self).__init__(**kwargs)
        assert self._dtype == self._data.dtype

    def replace(self, **kwargs):
        if "data" in kwargs and "dtype" not in kwargs:
            kwargs["dtype"] = kwargs["data"].dtype
        return super(NumericalColumn, self).replace(**kwargs)

    def serialize(self):
        header, frames = super(NumericalColumn, self).serialize()
        header["type"] = pickle.dumps(type(self))
        header["dtype"] = self._dtype.str
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        data, mask = super(NumericalColumn, cls).deserialize(header, frames)
        dtype = header["dtype"]
        col = cls(
            data=data, mask=mask, null_count=header["null_count"], dtype=dtype
        )
        return col

    def binary_operator(self, binop, rhs, reflect=False):
        if isinstance(rhs, NumericalColumn) or np.isscalar(rhs):
            out_dtype = np.result_type(self.dtype, rhs.dtype)
            return numeric_column_binop(
                lhs=self,
                rhs=rhs,
                op=binop,
                out_dtype=out_dtype,
                reflect=reflect,
            )
        else:
            msg = "{!r} operator not supported between {} and {}"
            raise TypeError(msg.format(binop, type(self), type(rhs)))

    def unary_operator(self, unaryop):
        return numeric_column_unaryop(self, op=unaryop)

    def unordered_compare(self, cmpop, rhs):
        return numeric_column_compare(self, rhs, op=cmpop)

    def ordered_compare(self, cmpop, rhs):
        return numeric_column_compare(self, rhs, op=cmpop)

    def _apply_scan_op(self, op):
        out_col = columnops.column_empty_like_same_mask(self, dtype=self.dtype)
        cpp_reduce.apply_scan(self, out_col, op, inclusive=True)
        return out_col

    def normalize_binop_value(self, other):
        other_dtype = np.min_scalar_type(other)
        if other_dtype.kind in "biuf":
            other_dtype = np.promote_types(self.dtype, other_dtype)
            if other_dtype == np.dtype("float16"):
                other = np.dtype("float32").type(other)
                other_dtype = other.dtype
            if other_dtype.kind in "u":
                other_dtype = utils.min_signed_type(other)
            if np.isscalar(other):
                other = np.dtype(other_dtype).type(other)
                return other
            else:
                ary = utils.scalar_broadcast_to(
                    other, shape=len(self), dtype=other_dtype
                )
                return self.replace(data=Buffer(ary), dtype=ary.dtype)
        else:
            raise TypeError("cannot broadcast {}".format(type(other)))

    def as_string_column(self, dtype, **kwargs):
        from cudf.dataframe import string

        if len(self) > 0:
            if self.dtype in (np.dtype("int8"), np.dtype("int16")):
                dev_array = self.astype("int32", **kwargs).data.mem
            else:
                dev_array = self.data.mem
            dev_ptr = get_ctype_ptr(dev_array)
            null_ptr = None
            if self.mask is not None:
                null_ptr = get_ctype_ptr(self.mask.mem)
            kwargs = {"count": len(self), "nulls": null_ptr, "bdevmem": True}
            data = string._numeric_to_str_typecast_functions[
                np.dtype(dev_array.dtype)
            ](dev_ptr, **kwargs)
        else:
            data = []
        return string.StringColumn(data=data)

    def as_datetime_column(self, dtype, **kwargs):
        from cudf.dataframe import datetime
        import cudf.bindings.typecast as typecast

        return self.view(
            datetime.DatetimeColumn,
            dtype=dtype,
            data=typecast.apply_cast(self, dtype=np.dtype(dtype)).data,
        )

    def as_numerical_column(self, dtype, **kwargs):
        import cudf.bindings.typecast as typecast

        return self.replace(
            data=typecast.apply_cast(self, dtype).data, dtype=np.dtype(dtype)
        )

    def sort_by_values(self, ascending=True, na_position="last"):
        sort_inds = get_sorted_inds(self, ascending, na_position)
        col_keys = cpp_copying.apply_gather(self, sort_inds.data.mem)
        col_inds = self.replace(
            data=sort_inds.data,
            mask=sort_inds.mask,
            dtype=sort_inds.data.dtype,
        )
        return col_keys, col_inds

    def to_pandas(self, index=None):
        if self.null_count > 0 and self.dtype == np.bool:
            # Boolean series in Pandas that contains None/NaN is of dtype
            # `np.object`, which is not natively supported in GDF.
            ret = self.astype(np.int8).to_array(fillna=-1)
            ret = pd.Series(ret, index=index)
            ret = ret.where(ret >= 0, other=None)
            ret.replace(to_replace=1, value=True, inplace=True)
            ret.replace(to_replace=0, value=False, inplace=True)
            return ret
        else:
            return pd.Series(self.to_array(fillna="pandas"), index=index)

    def to_arrow(self):
        mask = None
        if self.has_null_mask:
            mask = pa.py_buffer(self.nullmask.mem.copy_to_host())
        data = pa.py_buffer(self.data.mem.copy_to_host())
        pa_dtype = np_to_pa_dtype(self.dtype)
        out = pa.Array.from_buffers(
            type=pa_dtype,
            length=len(self),
            buffers=[mask, data],
            null_count=self.null_count,
        )
        if self.dtype == np.bool:
            return out.cast(pa.bool_())
        else:
            return out

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

    def all(self):
        return bool(self.min(dtype=np.bool_))

    def any(self):
        if self.valid_count == 0:
            return False
        return bool(self.max(dtype=np.bool_))

    def min(self, dtype=None):
        return cpp_reduce.apply_reduce("min", self, dtype=dtype)

    def max(self, dtype=None):
        return cpp_reduce.apply_reduce("max", self, dtype=dtype)

    def sum(self, dtype=None):
        return cpp_reduce.apply_reduce("sum", self, dtype=dtype)

    def product(self, dtype=None):
        return cpp_reduce.apply_reduce("product", self, dtype=dtype)

    def mean(self, dtype=np.float64):
        return cpp_reduce.apply_reduce("mean", self, dtype=dtype)

    def var(self, ddof=1, dtype=np.float64):
        return cpp_reduce.apply_reduce("var", self, dtype=dtype, ddof=ddof)

    def std(self, ddof=1, dtype=np.float64):
        return cpp_reduce.apply_reduce("std", self, dtype=dtype, ddof=ddof)

    def sum_of_squares(self, dtype=None):
        return cpp_reduce.apply_reduce("sum_of_squares", self, dtype=dtype)

    def round(self, decimals=0):
        data = Buffer(cudautils.apply_round(self.data.mem, decimals))
        return self.replace(data=data)

    def applymap(self, udf, out_dtype=None):
        """Apply a elemenwise function to transform the values in the Column.

        Parameters
        ----------
        udf : function
            Wrapped by numba jit for call on the GPU as a device function.
        out_dtype  : numpy.dtype; optional
            The dtype for use in the output.
            By default, use the same dtype as *self.dtype*.

        Returns
        -------
        result : Column
            The mask is preserved.
        """
        if out_dtype is None:
            out_dtype = self.dtype
        out = columnops.column_applymap(
            udf=udf, column=self, out_dtype=out_dtype
        )
        return self.replace(data=out, dtype=out_dtype)

    def default_na_value(self):
        """Returns the default NA value for this column
        """
        dkind = self.dtype.kind
        if dkind == "f":
            return self.dtype.type(np.nan)
        elif dkind in "iu":
            return -1
        elif dkind == "b":
            return False
        else:
            raise TypeError(
                "numeric column of {} has no NaN value".format(self.dtype)
            )

    def find_and_replace(self, to_replace, replacement, all_nan):
        """
        Return col with *to_replace* replaced with *value*.
        """
        to_replace_col = columnops.as_column(to_replace)
        replacement_dtype = self.dtype if all_nan else None
        replacement_col = columnops.as_column(
            replacement, dtype=replacement_dtype
        )
        replaced = self.copy()
        to_replace_col, replacement_col, replaced = numeric_normalize_types(
            to_replace_col, replacement_col, replaced
        )
        output = cpp_replace.replace(replaced, to_replace_col, replacement_col)
        return output

    def fillna(self, fill_value, inplace=False):
        """
        Fill null values with *fill_value*
        """
        if np.isscalar(fill_value):
            # castsafely to the same dtype as self
            fill_value_casted = self.dtype.type(fill_value)
            if not np.isnan(fill_value) and (fill_value_casted != fill_value):
                raise TypeError(
                    "Cannot safely cast non-equivalent {} to {}".format(
                        type(fill_value).__name__, self.dtype.name
                    )
                )
            fill_value = fill_value_casted
        else:
            fill_value = columnops.as_column(fill_value, nan_as_null=False)
            # cast safely to the same dtype as self
            if is_integer_dtype(self.dtype):
                fill_value = safe_cast_to_int(fill_value, self.dtype)
            else:
                fill_value = fill_value.astype(self.dtype)
        result = cpp_replace.apply_replace_nulls(self, fill_value)
        return self._mimic_inplace(result, inplace)

    def find_first_value(self, value):
        """
        Returns offset of first value that matches. For monotonic
        columns, returns the offset of the first larger value.
        """
        found = cudautils.find_first(self.data.mem, value)
        if found == -1 and self.is_monotonic:
            if value < self.min():
                found = 0
            elif value > self.max():
                found = len(self)
            else:
                found = cudautils.find_first(
                    self.data.mem, value, compare="gt"
                )
                if found == -1:
                    raise ValueError("value not found")
        elif found == -1:
            raise ValueError("value not found")
        return found

    def find_last_value(self, value):
        """
        Returns offset of last value that matches. For monotonic
        columns, returns the offset of the last smaller value.
        """
        found = cudautils.find_last(self.data.mem, value)
        if found == -1 and self.is_monotonic:
            if value < self.min():
                found = -1
            elif value > self.max():
                found = len(self) - 1
            else:
                found = cudautils.find_last(self.data.mem, value, compare="lt")
                if found == -1:
                    raise ValueError("value not found")
        elif found == -1:
            raise ValueError("value not found")
        return found

    def searchsorted(self, value, side="left"):
        value_col = columnops.as_column(value)
        return cpp_search.search_sorted(self, value_col, side)

    @property
    def is_monotonic_increasing(self):
        if not hasattr(self, "_is_monotonic_increasing"):
            self._is_monotonic_increasing = numeric_column_binop(
                self[1:], self[:-1], "ge", "bool"
            ).all()
        return self._is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        if not hasattr(self, "_is_monotonic_decreasing"):
            self._is_monotonic_decreasing = numeric_column_binop(
                self[1:], self[:-1], "le", "bool"
            ).all()
        return self._is_monotonic_decreasing


def numeric_column_binop(lhs, rhs, op, out_dtype, reflect=False):
    if reflect:
        lhs, rhs = rhs, lhs
    nvtx_range_push("CUDF_BINARY_OP", "orange")
    # Allocate output
    masked = False
    name = None
    if np.isscalar(lhs):
        masked = rhs.has_null_mask
        row_count = len(rhs)
        name = rhs.name
    elif np.isscalar(rhs):
        masked = lhs.has_null_mask
        row_count = len(lhs)
        name = lhs.name
    else:
        masked = lhs.has_null_mask or rhs.has_null_mask
        row_count = len(lhs)

    is_op_comparison = op in ["lt", "gt", "le", "ge", "eq", "ne"]

    out = columnops.column_empty(row_count, dtype=out_dtype, masked=masked)
    # Call and fix null_count
    null_count = cpp_binops.apply_op(lhs, rhs, out, op)

    if is_op_comparison:
        out.fillna(op == "ne", inplace=True)
    else:
        out = out.replace(null_count=null_count)

    result = out.view(NumericalColumn, dtype=out_dtype, name=name)
    nvtx_range_pop()
    return result


def numeric_column_unaryop(operand, op):
    out = cpp_unaryops.apply_unary_op(operand, op)
    return out.view(NumericalColumn, dtype=out.dtype)


def numeric_column_compare(lhs, rhs, op):
    return numeric_column_binop(lhs, rhs, op, out_dtype=np.bool_)


def numeric_normalize_types(*args):
    """Cast all args to a common type using numpy promotion logic
    """
    dtype = np.result_type(*[a.dtype for a in args])
    return [a.astype(dtype) for a in args]


def safe_cast_to_int(col, dtype):
    """
    Cast given NumericalColumn to given integer dtype safely.
    """
    assert is_integer_dtype(dtype)

    if col.dtype == dtype:
        return col

    new_col = col.astype(dtype)
    if new_col.unordered_compare("eq", col).all():
        return new_col
    else:
        raise TypeError(
            "Cannot safely cast non-equivalent {} to {}".format(
                col.dtype.type.__name__, np.dtype(dtype).type.__name__
            )
        )


def column_hash_values(column0, *other_columns, initial_hash_values=None):
    """Hash all values in the given columns.
    Returns a new NumericalColumn[int32]
    """
    columns = [column0] + list(other_columns)
    buf = Buffer(rmm.device_array(len(column0), dtype=np.int32))
    result = NumericalColumn(data=buf, dtype=buf.dtype)
    if initial_hash_values:
        initial_hash_values = rmm.to_device(initial_hash_values)
    cpp_hash.hash_columns(columns, result, initial_hash_values)
    return result


def digitize(column, bins, right=False):
    """Return the indices of the bins to which each value in column belongs.

    Parameters
    ----------
    column : Column
        Input column.
    bins : np.array
        1-D monotonically increasing array of bins with same type as `column`.
    right : bool
        Indicates whether interval contains the right or left bin edge.

    Returns
    -------
    A device array containing the indices
    """
    assert column.dtype == bins.dtype
    bins_buf = Buffer(rmm.to_device(bins))
    bin_col = NumericalColumn(data=bins_buf, dtype=bins.dtype)
    return cpp_sort.digitize(column, bin_col, right)
