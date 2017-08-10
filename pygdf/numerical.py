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


class NumericalColumn(columnops.ColumnOps):
    def __init__(self, **kwargs):
        super(NumericalColumn, self).__init__(**kwargs)
        assert self._dtype == self._data.dtype

    def binary_operator(self, binop, rhs):
        if isinstance(rhs, NumericalColumn):
            op = _binary_impl[binop]
            return numeric_column_binop(lhs=self, rhs=rhs, op=op,
                                        out_dtype=self.dtype)
        else:
            return NotImplemented

    def unary_operator(self, unaryop):
        return numeric_column_unaryop(self, op=_unary_impl[unaryop],
                                      out_dtype=self.dtype)

    def unordered_compare(self, cmpop, rhs):
        return numeric_column_compare(self, rhs, op=_unordered_impl[cmpop])

    def ordered_compare(self, cmpop, rhs):
        return numeric_column_compare(self, rhs, op=_ordered_impl[cmpop])

    def normalize_compare_value(self, other):
        if np.min_scalar_type(other).kind in 'biuf':
            ary = utils.scalar_broadcast_to(other, shape=len(self))
            return self.replace(data=Buffer(ary), dtype=ary.dtype)
        else:
            raise TypeError('cannot broadcast {}'.format(type(other)))

    @property
    def stats(self):
        return ColumnStats(self)

    def astype(self, dtype):
        if self.dtype == dtype:
            return self
        else:
            col = self.replace(data=self.data.astype(dtype),
                               dtype=dtype)
            return col

    def sort_by_values(self, ascending):
        if self.mask:
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

    def all(self):
        return self.stats.min() == True


class ColumnStats(object):
    def __init__(self, column):
        self._column = column

    def min(self):
        return _gdf.apply_reduce(libgdf.gdf_min_generic, self._column)

    def max(self):
        return _gdf.apply_reduce(libgdf.gdf_max_generic, self._column)

    def sum(self):
        dt = np.promote_types('i8', self._column.dtype)
        x = self._column.astype(dt)
        return _gdf.apply_reduce(libgdf.gdf_sum_generic, x)

    def mean(self):
        return self.sum().astype('f8') / self._column.valid_count

    def mean_var(self):
        x = self._column.astype('f8')
        mu = x.stats.mean()
        n = x.valid_count
        asum = _gdf.apply_reduce(libgdf.gdf_sum_squared_generic, x)
        var = asum / n - mu ** 2
        return mu, var


def numeric_column_binop(lhs, rhs, op, out_dtype):
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
