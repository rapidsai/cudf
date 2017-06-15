import numpy as np

from libgdf_cffi import libgdf

from .series_impl import SeriesImpl
from .dataframe import Series
from . import _gdf


unordered_impl = {
    'eq': libgdf.gdf_eq_generic,
    'ne': libgdf.gdf_ne_generic,
}

ordered_impl = {
    'lt': libgdf.gdf_lt_generic,
    'le': libgdf.gdf_le_generic,
    'gt': libgdf.gdf_gt_generic,
    'ge': libgdf.gdf_ge_generic,
}


class NumericalSeriesImpl(SeriesImpl):
    def __init__(self, dtype):
        super(NumericalSeriesImpl, self).__init__(dtype)

    def element_to_str(self, value):
        return str(value)

    def unordered_compare(self, cmpop, lhs, rhs):
        if not isinstance(rhs, Series):
            return NotImplemented
        return self.compare(lhs, rhs, fn=unordered_impl[cmpop])

    def ordered_compare(self, cmpop, lhs, rhs):
        if not isinstance(rhs, Series):
            return NotImplemented
        return self.compare(lhs, rhs, fn=ordered_impl[cmpop])

    #
    # Helpers
    #

    def compare(self, lhs, rhs, fn):
        """
        Internal util to call a comparison operator *fn*
        comparing *lhs* and *rhs*.  Return the output Series.
        The output dtype is always `np.bool_`.
        """
        return self._call_binop(lhs, rhs, fn, np.bool_)

    #
    # Internals
    #

    def _call_binop(self, lhs, rhs, fn, out_dtype):
        """
        Internal util to call a binary operator *fn* on operands *lhs*
        and *rhs* with output dtype *out_dtype*.  Returns the output
        Series.
        """
        # Allocate output series
        needs_mask = lhs.has_null_mask or rhs.has_null_mask
        out = lhs._empty_like(dtype=out_dtype, has_mask=needs_mask,
                              impl=NumericalSeriesImpl(out_dtype))
        # Call and fix null_count
        out._null_count = _gdf.apply_binaryop(fn, lhs, rhs, out)
        return out
