# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import print_function, division

import numpy as np
import pandas as pd
import pyarrow as pa

from libgdf_cffi import libgdf
from librmm_cffi import librmm as rmm

from . import columnops, datetime
from cudf.utils import cudautils, utils
from cudf import _gdf
from .buffer import Buffer
from cudf.comm.serialize import register_distributed_serializer
from cudf._gdf import nvtx_range_push, nvtx_range_pop
from cudf._sort import get_sorted_inds

import cudf.bindings.reduce as cpp_reduce
import cudf.bindings.replace as cpp_replace

# Operator mappings

#   Unordered comparators
_unordered_impl = {
    'eq': libgdf.gdf_eq_generic,
    'ne': libgdf.gdf_ne_generic,
}

#   Ordered comparators
_ordered_impl = {
    'lt': libgdf.gdf_lt_generic,
    'le': libgdf.gdf_le_generic,
    'gt': libgdf.gdf_gt_generic,
    'ge': libgdf.gdf_ge_generic,
}

#   Binary operators
_binary_impl = {
    'add': libgdf.gdf_add_generic,
    'sub': libgdf.gdf_sub_generic,
    'mul': libgdf.gdf_mul_generic,
    'floordiv': libgdf.gdf_floordiv_generic,
    'truediv': libgdf.gdf_div_generic,
}

#   Unary operators
_unary_impl = {
    'ceil': libgdf.gdf_ceil_generic,
    'floor': libgdf.gdf_floor_generic,
}


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

    def binary_operator(self, binop, rhs):
        if isinstance(rhs, NumericalColumn):
            op = _binary_impl[binop]
            lhs, rhs = numeric_normalize_types(self, rhs)
            return numeric_column_binop(lhs=lhs, rhs=rhs, op=op,
                                        out_dtype=lhs.dtype)
        else:
            msg = "{!r} operator not supported between {} and {}"
            raise TypeError(msg.format(binop, type(self), type(rhs)))

    def unary_operator(self, unaryop):
        return numeric_column_unaryop(self, op=_unary_impl[unaryop],
                                      out_dtype=self.dtype)

    def unordered_compare(self, cmpop, rhs):
        lhs, rhs = numeric_normalize_types(self, rhs)
        return numeric_column_compare(lhs, rhs, op=_unordered_impl[cmpop])

    def ordered_compare(self, cmpop, rhs):
        lhs, rhs = numeric_normalize_types(self, rhs)
        return numeric_column_compare(lhs, rhs, op=_ordered_impl[cmpop])

    def normalize_binop_value(self, other):
        other_dtype = np.min_scalar_type(other)
        if other_dtype.kind in 'biuf':
            other_dtype = np.promote_types(self.dtype, other_dtype)
            ary = utils.scalar_broadcast_to(other, shape=len(self),
                                            dtype=other_dtype)
            return self.replace(data=Buffer(ary), dtype=ary.dtype)
        else:
            raise TypeError('cannot broadcast {}'.format(type(other)))

    def astype(self, dtype):
        if self.dtype == dtype:
            return self
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
        col_keys = cudautils.gather(data=self.data.mem,
                                    index=sort_inds.data.mem)
        mask = None
        if self.mask:
            mask = self._get_mask_as_column()\
                .take(sort_inds.data.to_gpu_array()).as_mask()
            mask = Buffer(mask)
        col_keys = self.replace(data=Buffer(col_keys),
                                mask=mask,
                                null_count=self.null_count,
                                dtype=self.dtype)
        col_inds = self.replace(data=sort_inds.data,
                                mask=sort_inds.mask,
                                dtype=sort_inds.data.dtype)
        return col_keys, col_inds

    def to_pandas(self, index=None):
        return pd.Series(self.to_array(fillna='pandas'), index=index)

    def to_arrow(self):
        mask = None
        if self.has_null_mask:
            mask = pa.py_buffer(self.nullmask.mem.copy_to_host())
        data = pa.py_buffer(self.data.mem.copy_to_host())
        pa_dtype = _gdf.np_to_pa_dtype(self.dtype)
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
        if method is not 'sort':
            msg = 'non sort based unique() not implemented yet'
            raise NotImplementedError(msg)
        segs, sortedvals = self._unique_segments()
        # gather result
        out = cudautils.gather(data=sortedvals, index=segs)
        return self.replace(data=Buffer(out), mask=None)

    def unique_count(self, method='sort'):
        if method is not 'sort':
            msg = 'non sort based unique_count() not implemented yet'
            raise NotImplementedError(msg)
        segs, _ = self._unique_segments()
        return len(segs)

    def value_counts(self, method='sort'):
        if method is not 'sort':
            msg = 'non sort based value_count() not implemented yet'
            raise NotImplementedError(msg)
        segs, sortedvals = self._unique_segments()
        # Return both values and their counts
        out1 = cudautils.gather(data=sortedvals, index=segs)
        out2 = cudautils.value_count(segs, len(sortedvals))
        out_vals = self.replace(data=Buffer(out1), mask=None)
        out_counts = NumericalColumn(data=Buffer(out2), dtype=np.intp)
        return out_vals, out_counts

    def all(self):
        return bool(self.min())

    def min(self):
        return cpp_reduce.apply_reduce('min', self)

    def max(self):
        return cpp_reduce.apply_reduce('max', self)

    def sum(self):
        return cpp_reduce.apply_reduce('sum', self)

    def product(self):
        return cpp_reduce.apply_reduce('product', self)

    def mean(self):
        return self.sum().astype('f8') / self.valid_count

    def mean_var(self, ddof=1):
        x = self.astype('f8')
        mu = x.mean()
        n = x.valid_count
        asum = x.sum_of_squares()
        div = n - ddof
        var = asum / div - (mu ** 2) * n / div
        return mu, var

    def sum_of_squares(self):
        return cpp_reduce.apply_reduce('sum_of_squares', self)

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
            return np.nan
        elif dkind in 'iu':
            return -1
        else:
            raise TypeError(
                "numeric column of {} has no NaN value".format(self.dtype))

    def join(self, other, how='left', return_indexers=False, method='sort'):

        # Single column join using sort-based implementation
        if method == 'sort' or how == 'outer':
            return self._sortjoin(other=other, how=how,
                                  return_indexers=return_indexers)
        elif method == 'hash':
            # Get list of columns from self with left_on and
            # from other with right_on
            return self._hashjoin(other=other, how=how,
                                  return_indexers=return_indexers)
        else:
            raise ValueError('Unsupported join method')

    def _hashjoin(self, other, how='left', return_indexers=False):

        from .series import Series

        if not self.is_type_equivalent(other):
            raise TypeError('*other* is not compatible')

        with _gdf.apply_join(
                [self], [other], how=how, method='hash') as (lidx, ridx):
            if lidx.size > 0:
                raw_index = cudautils.gather_joined_index(
                        self.to_gpu_array(),
                        other.to_gpu_array(),
                        lidx,
                        ridx,
                        )
                buf_index = Buffer(raw_index)
            else:
                buf_index = Buffer.null(dtype=self.dtype)

            joined_index = self.replace(data=buf_index)

            if return_indexers:
                def gather(idxrange, idx):
                    mask = (Series(idx) != -1).as_mask()
                    return idxrange.take(idx).set_mask(mask).fillna(-1)

                if len(joined_index) > 0:
                    indexers = (
                            gather(Series(range(0, len(self))), lidx),
                            gather(Series(range(0, len(other))), ridx),
                            )
                else:
                    indexers = (
                            Series(Buffer.null(dtype=np.intp)),
                            Series(Buffer.null(dtype=np.intp))
                            )
                return joined_index, indexers
            else:
                return joined_index
        # return

    def _sortjoin(self, other, how='left', return_indexers=False):
        """Join with another column.

        When the column is a index, set *return_indexers* to obtain
        the indices for shuffling the remaining columns.
        """
        from .series import Series

        if not self.is_type_equivalent(other):
            raise TypeError('*other* is not compatible')

        lkey, largsort = self.sort_by_values(True)
        rkey, rargsort = other.sort_by_values(True)
        with _gdf.apply_join(
                [lkey], [rkey], how=how, method='sort') as (lidx, ridx):
            if lidx.size > 0:
                raw_index = cudautils.gather_joined_index(
                        lkey.to_gpu_array(),
                        rkey.to_gpu_array(),
                        lidx,
                        ridx,
                        )
                buf_index = Buffer(raw_index)
            else:
                buf_index = Buffer.null(dtype=self.dtype)

            joined_index = lkey.replace(data=buf_index)

            if return_indexers:
                def gather(idxrange, idx):
                    mask = (Series(idx) != -1).as_mask()
                    return idxrange.take(idx).set_mask(mask).fillna(-1)

                if len(joined_index) > 0:
                    indexers = (
                            gather(Series(largsort), lidx),
                            gather(Series(rargsort), ridx),
                            )
                else:
                    indexers = (
                            Series(Buffer.null(dtype=np.intp)),
                            Series(Buffer.null(dtype=np.intp))
                            )
                return joined_index, indexers
            else:
                return joined_index

    def find_and_replace(self, to_replace, value):
        """
        Return col with *to_replace* replaced with *value*.
        """
        to_replace_col = columnops.as_column(to_replace)
        value_col = columnops.as_column(value)
        replaced = self.copy()
        to_replace_col, value_col, replaced = numeric_normalize_types(
               to_replace_col, value_col, replaced)
        cpp_replace.replace(replaced, to_replace_col, value_col)
        return replaced


def numeric_column_binop(lhs, rhs, op, out_dtype):
    if lhs.dtype != rhs.dtype:
        raise TypeError('{} != {}'.format(lhs.dtype, rhs.dtype))

    nvtx_range_push("PYGDF_BINARY_OP", "orange")
    # Allocate output
    masked = lhs.has_null_mask or rhs.has_null_mask
    out = columnops.column_empty_like(lhs, dtype=out_dtype, masked=masked)
    # Call and fix null_count
    null_count = _gdf.apply_binaryop(op, lhs, rhs, out)
    out = out.replace(null_count=null_count)
    result = out.view(NumericalColumn, dtype=out_dtype)
    nvtx_range_pop()
    return result


def numeric_column_unaryop(operand, op, out_dtype):
    out = columnops.column_empty_like_same_mask(operand, dtype=out_dtype)
    _gdf.apply_unaryop(op, operand, out)
    return out.view(NumericalColumn, dtype=out_dtype)


def numeric_column_compare(lhs, rhs, op):
    return numeric_column_binop(lhs, rhs, op, out_dtype=np.bool_)


def numeric_normalize_types(*args):
    """Cast all args to a common type using numpy promotion logic
    """
    dtype = np.result_type(*[a.dtype for a in args])
    return [a.astype(dtype) for a in args]


def column_hash_values(column0, *other_columns):
    """Hash all values in the given columns.
    Returns a new NumericalColumn[int32]
    """
    columns = [column0] + list(other_columns)
    buf = Buffer(rmm.device_array(len(column0), dtype=np.int32))
    result = NumericalColumn(data=buf, dtype=buf.dtype)
    _gdf.hash_columns(columns, result)
    return result


register_distributed_serializer(NumericalColumn)
