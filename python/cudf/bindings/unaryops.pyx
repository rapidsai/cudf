# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from .cudf_cpp cimport *
from .cudf_cpp import *

from librmm_cffi import librmm as rmm

from libc.stdlib cimport free


_MATH_OP = {}
_MATH_OP['sin'] = GDF_SIN
_MATH_OP['cos'] = GDF_COS
_MATH_OP['tan'] = GDF_TAN
_MATH_OP['asin'] = GDF_ARCSIN
_MATH_OP['acos'] = GDF_ARCCOS
_MATH_OP['atan'] = GDF_ARCTAN
_MATH_OP['exp'] = GDF_EXP
_MATH_OP['log'] = GDF_LOG
_MATH_OP['sqrt'] = GDF_SQRT
_MATH_OP['ceil'] = GDF_CEIL
_MATH_OP['floor'] = GDF_FLOOR


def apply_math_op(incol, outcol, op):
    """
      Call Unary math ops.
    """

    check_gdf_compatibility(incol)
    check_gdf_compatibility(outcol)
    
    cdef gdf_column* c_incol = column_view_from_column(incol)
    cdef gdf_column* c_outcol = column_view_from_column(outcol)

    cdef gdf_error result
    cdef gdf_unary_math_op c_op = _MATH_OP[op]
    with nogil:    
        result = gdf_unary_math(
            <gdf_column*>c_incol,
            <gdf_column*>c_outcol,
            c_op)
    
    free(c_incol)
    free(c_outcol)

    check_gdf_error(result)
