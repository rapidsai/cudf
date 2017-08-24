from __future__ import print_function, division

import numpy as np
import pandas as pd

from libgdf_cffi import libgdf

from . import _gdf, columnops, utils, cudautils
from .buffer import Buffer


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

    def normalize_compare_value(self, other):
        return self.normalize_binop_value(other)

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
        else:
            col = self.replace(data=self.data.astype(dtype),
                               dtype=np.dtype(dtype))
            return col

    def sort_by_values(self, ascending):
        if self.has_null_mask:
            raise ValueError('masked array not supported')
        # Clone data buffer as the key
        col_keys = self.replace(data=self.data.copy())
        # Create new array for the positions
        inds = Buffer(cudautils.arange(len(self)))
        col_inds = self.replace(data=inds, dtype=inds.dtype)
        _gdf.apply_sort(col_keys, col_inds, ascending=ascending)
        return col_keys, col_inds

    def to_pandas(self, index=None):
        return pd.Series(self.to_array(fillna='pandas'), index=index)

    def unique_k(self, k):
        # make dense column
        densecol = self.replace(data=self.to_dense_buffer(), mask=None)
        # sort the column
        sortcol, _ = densecol.sort_by_values(ascending=True)
        # find segments
        sortedvals = sortcol.to_gpu_array()
        segs = cudautils.find_segments(sortedvals)
        # TODO: we can now support unlimited number of unique values
        #       thus, we don't need to set the limit
        if segs.size > k:
            raise ValueError('too many unique value')
        # gather result
        out = cudautils.gather(data=sortedvals, index=segs)
        return self.replace(data=Buffer(out), mask=None)

    def all(self):
        return bool(self.min())

    def min(self):
        return _gdf.apply_reduce(libgdf.gdf_min_generic, self)

    def max(self):
        return _gdf.apply_reduce(libgdf.gdf_max_generic, self)

    def sum(self):
        dt = np.promote_types('i8', self.dtype)
        x = self.astype(dt)
        return _gdf.apply_reduce(libgdf.gdf_sum_generic, x)

    def mean(self):
        return self.sum().astype('f8') / self.valid_count

    def mean_var(self):
        x = self.astype('f8')
        mu = x.mean()
        n = x.valid_count
        asum = _gdf.apply_reduce(libgdf.gdf_sum_squared_generic, x)
        var = asum / n - mu ** 2
        return mu, var

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
        return self.replace(data=out)


def numeric_column_binop(lhs, rhs, op, out_dtype):
    if lhs.dtype != rhs.dtype:
        raise TypeError('{} != {}'.format(lhs.dtype, rhs.dtype))
    # Allocate output
    masked = lhs.has_null_mask or rhs.has_null_mask
    out = columnops.column_empty_like(lhs, dtype=out_dtype, masked=masked)
    # Call and fix null_count
    null_count = _gdf.apply_binaryop(op, lhs, rhs, out)
    out = out.replace(null_count=null_count)
    return out.view(NumericalColumn, dtype=out_dtype)


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
