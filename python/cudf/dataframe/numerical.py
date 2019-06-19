# Copyright (c) 2018-2019, NVIDIA CORPORATION.

from __future__ import print_function, division

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.types import is_integer_dtype

from librmm_cffi import librmm as rmm

from cudf.dataframe import columnops, datetime, string
from cudf.utils import cudautils, utils
from cudf.dataframe.buffer import Buffer
from cudf.comm.serialize import register_distributed_serializer
from cudf.bindings.nvtx import nvtx_range_push, nvtx_range_pop
from cudf.bindings.cudf_cpp import np_to_pa_dtype
from cudf._sort import get_sorted_inds

import cudf.bindings.reduce as cpp_reduce
import cudf.bindings.replace as cpp_replace
import cudf.bindings.binops as cpp_binops
import cudf.bindings.sort as cpp_sort
import cudf.bindings.unaryops as cpp_unaryops
import cudf.bindings.copying as cpp_copying
import cudf.bindings.hash as cpp_hash
from cudf.bindings.cudf_cpp import get_ctype_ptr


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
        if 'data' in kwargs and 'dtype' not in kwargs:
            kwargs['dtype'] = kwargs['data'].dtype
        return super(NumericalColumn, self).replace(**kwargs)

    def serialize(self, serialize):
        header, frames = super(NumericalColumn, self).serialize(serialize)
        assert 'dtype' not in header
        header['dtype'] = serialize(self._dtype)
        return header, frames

    @classmethod
    def deserialize(cls, deserialize, header, frames):
        data, mask = cls._deserialize_data_mask(deserialize, header, frames)
        col = cls(data=data, mask=mask, null_count=header['null_count'],
                  dtype=deserialize(*header['dtype']))
        return col

    def binary_operator(self, binop, rhs, reflect=False):
        if isinstance(rhs, NumericalColumn) or np.isscalar(rhs):
            out_dtype = np.result_type(self.dtype, rhs.dtype)
            return numeric_column_binop(
                lhs=self,
                rhs=rhs,
                op=binop,
                out_dtype=out_dtype,
                reflect=reflect
            )
        else:
            msg = "{!r} operator not supported between {} and {}"
            raise TypeError(msg.format(binop, type(self), type(rhs)))

    def unary_operator(self, unaryop):
        return numeric_column_unaryop(self, op=unaryop,
                                      out_dtype=self.dtype)

    def unary_logic_op(self, unaryop):
        return numeric_column_unaryop(self, op=unaryop,
                                      out_dtype=np.bool_)

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
        if other_dtype.kind in 'biuf':
            other_dtype = np.promote_types(self.dtype, other_dtype)

            if np.isscalar(other):
                other = np.dtype(other_dtype).type(other)
                return other
            else:
                ary = utils.scalar_broadcast_to(
                    other,
                    shape=len(self),
                    dtype=other_dtype
                )
                return self.replace(data=Buffer(ary), dtype=ary.dtype)
        else:
            raise TypeError('cannot broadcast {}'.format(type(other)))

    def astype(self, dtype):
        if self.dtype == dtype:
            return self

        elif (dtype == np.dtype('object') or
              np.issubdtype(dtype, np.dtype('U').type)):
            if len(self) > 0:
                if self.dtype in (np.dtype('int8'), np.dtype('int16')):
                    dev_array = self.astype('int32').data.mem
                else:
                    dev_array = self.data.mem
                dev_ptr = get_ctype_ptr(dev_array)
                null_ptr = None
                if self.mask is not None:
                    null_ptr = get_ctype_ptr(self.mask.mem)
                kwargs = {
                    'count': len(self),
                    'nulls': null_ptr,
                    'bdevmem': True
                }
                data = string._numeric_to_str_typecast_functions[
                    np.dtype(dev_array.dtype)
                ](dev_ptr, **kwargs)

            else:
                data = []

            return string.StringColumn(data=data)

        elif np.issubdtype(dtype, np.datetime64):
            return self.astype('int64').view(
                datetime.DatetimeColumn,
                dtype=dtype,
                data=self.data.astype(dtype)
            )

        else:
            col = self.replace(data=self.data.astype(dtype),
                               dtype=np.dtype(dtype))
            return col

    def sort_by_values(self, ascending=True, na_position="last"):
        sort_inds = get_sorted_inds(self, ascending, na_position)
        col_keys = cpp_copying.apply_gather_column(self, sort_inds.data.mem)
        col_inds = self.replace(data=sort_inds.data,
                                mask=sort_inds.mask,
                                dtype=sort_inds.data.dtype)
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
            return pd.Series(self.to_array(fillna='pandas'), index=index)

    def to_arrow(self):
        mask = None
        if self.has_null_mask:
            mask = pa.py_buffer(self.nullmask.mem.copy_to_host())
        data = pa.py_buffer(self.data.mem.copy_to_host())
        pa_dtype = np_to_pa_dtype(self.dtype)
        out = pa.Array.from_buffers(
            type=pa_dtype,
            length=len(self),
            buffers=[
                mask,
                data
            ],
            null_count=self.null_count
        )
        if self.dtype == np.bool:
            return out.cast(pa.bool_())
        else:
            return out

    def _unique_segments(self):
        """ Common code for unique, unique_count and value_counts"""
        # make dense column
        densecol = self.replace(data=self.to_dense_buffer(), mask=None)
        # sort the column
        sortcol, _ = densecol.sort_by_values(ascending=True)
        # find segments
        sortedvals = sortcol.to_gpu_array()
        segs, begins = cudautils.find_segments(sortedvals)
        return segs, sortedvals

    def unique(self, method='sort'):
        # method variable will indicate what algorithm to use to
        # calculate unique, not used right now
        if method != 'sort':
            msg = 'non sort based unique() not implemented yet'
            raise NotImplementedError(msg)
        segs, sortedvals = self._unique_segments()
        # gather result
        out_col = cpp_copying.apply_gather_array(sortedvals, segs)
        return out_col

    def unique_count(self, method='sort', dropna=True):
        if method != 'sort':
            msg = 'non sort based unique_count() not implemented yet'
            raise NotImplementedError(msg)
        segs, _ = self._unique_segments()
        if dropna is False and self.null_count > 0:
            return len(segs)+1
        return len(segs)

    def value_counts(self, method='sort'):
        if method != 'sort':
            msg = 'non sort based value_count() not implemented yet'
            raise NotImplementedError(msg)
        segs, sortedvals = self._unique_segments()
        # Return both values and their counts
        out_vals = cpp_copying.apply_gather_array(sortedvals, segs)
        out2 = cudautils.value_count(segs, len(sortedvals))
        out_counts = NumericalColumn(data=Buffer(out2), dtype=np.intp)
        return out_vals, out_counts

    def all(self):
        return bool(self.min(dtype=np.bool_))

    def any(self):
        if self.valid_count == 0:
            return False
        return bool(self.max(dtype=np.bool_))

    def min(self, dtype=None):
        return cpp_reduce.apply_reduce('min', self, dtype=dtype)

    def max(self, dtype=None):
        return cpp_reduce.apply_reduce('max', self, dtype=dtype)

    def sum(self, dtype=None):
        return cpp_reduce.apply_reduce('sum', self, dtype=dtype)

    def product(self, dtype=None):
        return cpp_reduce.apply_reduce('product', self, dtype=dtype)

    def mean(self, dtype=np.float64):
        return np.float64(self.sum(dtype=dtype)) / self.valid_count

    def mean_var(self, ddof=1, dtype=np.float64):
        mu = self.mean(dtype=dtype)
        n = self.valid_count
        asum = np.float64(self.sum_of_squares(dtype=dtype))
        div = n - ddof
        var = asum / div - (mu ** 2) * n / div
        return mu, var

    def sum_of_squares(self, dtype=None):
        return cpp_reduce.apply_reduce('sum_of_squares', self, dtype=dtype)

    def round(self, decimals=0):
        mask = None
        if self.has_null_mask:
            mask = self.nullmask

        rounded = cudautils.apply_round(self.data.mem, decimals)
        return NumericalColumn(data=Buffer(rounded), mask=mask,
                               dtype=self.dtype)

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
        out = columnops.column_applymap(udf=udf, column=self,
                                        out_dtype=out_dtype)
        return self.replace(data=out, dtype=out_dtype)

    def default_na_value(self):
        """Returns the default NA value for this column
        """
        dkind = self.dtype.kind
        if dkind == 'f':
            return self.dtype.type(np.nan)
        elif dkind in 'iu':
            return -1
        elif dkind == 'b':
            return False
        else:
            raise TypeError(
                "numeric column of {} has no NaN value".format(self.dtype))

    def find_and_replace(self, to_replace, replacement, all_nan):
        """
        Return col with *to_replace* replaced with *value*.
        """
        to_replace_col = columnops.as_column(to_replace)
        replacement_dtype = self.dtype if all_nan else None
        replacement_col = columnops.as_column(replacement,
                                              dtype=replacement_dtype)
        replaced = self.copy()
        to_replace_col, replacement_col, replaced = numeric_normalize_types(
               to_replace_col, replacement_col, replaced)
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
        Returns offset of first value that matches
        """
        found = cudautils.find_first(
            self.data.mem,
            value)
        if found == -1:
            raise ValueError('value not found')
        return found

    def find_last_value(self, value):
        """
        Returns offset of last value that matches
        """
        found = cudautils.find_last(
            self.data.mem,
            value)
        if found == -1:
            raise ValueError('value not found')
        return found


def numeric_column_binop(lhs, rhs, op, out_dtype, reflect=False):
    if reflect:
        lhs, rhs = rhs, lhs
    nvtx_range_push("CUDF_BINARY_OP", "orange")
    # Allocate output
    masked = False
    if np.isscalar(lhs):
        masked = rhs.has_null_mask
        row_count = len(rhs)
    elif np.isscalar(rhs):
        masked = lhs.has_null_mask
        row_count = len(lhs)
    else:
        masked = lhs.has_null_mask or rhs.has_null_mask
        row_count = len(lhs)

    out = columnops.column_empty(row_count, dtype=out_dtype, masked=masked)
    # Call and fix null_count
    null_count = cpp_binops.apply_op(lhs, rhs, out, op)

    out = out.replace(null_count=null_count)
    result = out.view(NumericalColumn, dtype=out_dtype)
    nvtx_range_pop()
    return result


def numeric_column_unaryop(operand, op, out_dtype):
    out = columnops.column_empty_like_same_mask(operand, dtype=out_dtype)
    cpp_unaryops.apply_math_op(operand, out, op)
    return out.view(NumericalColumn, dtype=out_dtype)


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
    if new_col.unordered_compare('eq', col).all():
        return new_col
    else:
        raise TypeError("Cannot safely cast non-equivalent {} to {}".format(
            col.dtype.type.__name__,
            np.dtype(dtype).type.__name__))


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


register_distributed_serializer(NumericalColumn)
