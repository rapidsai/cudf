import numpy as np

from numba import cuda

from libgdf_cffi import libgdf

from .dataframe import Buffer
from .series_impl import SeriesImpl
from . import _gdf


_unordered_impl = {
    'eq': libgdf.gdf_eq_generic,
    'ne': libgdf.gdf_ne_generic,
}

_ordered_impl = {
    'lt': libgdf.gdf_lt_generic,
    'le': libgdf.gdf_le_generic,
    'gt': libgdf.gdf_gt_generic,
    'ge': libgdf.gdf_ge_generic,
}

_binary_impl = {
    'add': libgdf.gdf_add_generic,
    'sub': libgdf.gdf_sub_generic,
    'mul': libgdf.gdf_mul_generic,
    'floordiv': libgdf.gdf_floordiv_generic,
    'truediv': libgdf.gdf_div_generic,
}

_unary_impl = {
    'ceil': libgdf.gdf_ceil_generic,
    'floor': libgdf.gdf_floor_generic,
}


class NumericalSeriesImpl(SeriesImpl):
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
        needs_mask = lhs.has_null_mask or rhs.has_null_mask
        out = lhs._empty_like(dtype=out_dtype, has_mask=needs_mask,
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
        data = cuda.device_array_like(series.data.mem)
        out = series._copy_construct(dtype=out_dtype, buffer=Buffer(data),
                                     impl=NumericalSeriesImpl(out_dtype))
        _gdf.apply_unaryop(fn, series, out)
        return out
