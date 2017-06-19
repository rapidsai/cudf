import numpy as np

from libgdf_cffi import libgdf

from . import _gdf, series_impl, utils


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


class NumericalSeriesImpl(series_impl.SeriesImpl):
    """
    Implements operations for numerical Series.
    """
    def __init__(self, dtype):
        super(NumericalSeriesImpl, self).__init__(dtype)

    def element_to_str(self, value):
        return str(value)

    def binary_operator(self, binop, lhs, rhs):
        fn = _binary_impl[binop]
        return self._call_binop(lhs, rhs, fn, self.dtype)

    def unary_operator(self, unaryop, series):
        return self._call_unaryop(series, _unary_impl[unaryop], self.dtype)

    def unordered_compare(self, cmpop, lhs, rhs):
        return self._compare(lhs, rhs, fn=_unordered_impl[cmpop])

    def ordered_compare(self, cmpop, lhs, rhs):
        return self._compare(lhs, rhs, fn=_ordered_impl[cmpop])

    def normalize_compare_value(self, series, other):
        if np.min_scalar_type(other).kind in 'biuf':
            ary = utils.scalar_broadcast_to(other, shape=len(series))
            sr = series.from_any(ary)
            return sr
        return NotImplemented

    def element_indexing(self, series, index):
        return series_impl.element_indexing(series, index)

    #
    # Internals
    #

    def _compare(self, lhs, rhs, fn):
        """
        Internal util to call a comparison operator *fn*
        comparing *lhs* and *rhs*.  Return the output Series.
        The output dtype is always `np.bool_`.
        """
        return self._call_binop(lhs, rhs, fn, np.bool_)

    def _call_binop(self, lhs, rhs, fn, out_dtype):
        """
        Internal util to call a binary operator *fn* on operands *lhs*
        and *rhs* with output dtype *out_dtype*.  Returns the output
        Series.
        """
        # Allocate output series
        masked = lhs.has_null_mask or rhs.has_null_mask
        out = series_impl.empty_like(lhs, dtype=out_dtype, masked=masked,
                                     impl=NumericalSeriesImpl(out_dtype))
        # Call and fix null_count
        out._null_count = _gdf.apply_binaryop(fn, lhs, rhs, out)
        return out

    def _call_unaryop(self, series, fn, out_dtype):
        """
        Internal util to call a unary operator *fn* on operands *self* with
        output dtype *out_dtype*.  Returns the output Series.
        """
        # Allocate output series
        out = series_impl.empty_like_same_mask(series, dtype=out_dtype)
        _gdf.apply_unaryop(fn, series, out)
        return out
