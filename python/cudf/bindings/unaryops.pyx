# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from .cudf_cpp cimport *
from .cudf_cpp import *

from librmm_cffi import librmm as rmm

from libc.stdlib cimport free


_MATH_OP = {
    'sin'   : GDF_SIN,
    'cos'   : GDF_COS,
    'tan'   : GDF_TAN,
    'asin'  : GDF_ARCSIN,
    'acos'  : GDF_ARCCOS,
    'atan'  : GDF_ARCTAN,
    'exp'   : GDF_EXP,
    'log'   : GDF_LOG,
    'sqrt'  : GDF_SQRT,
    'ceil'  : GDF_CEIL,
    'floor' : GDF_FLOOR,
    'abs'   : GDF_ABS,
    'not'   : GDF_BIT_INVERT,
}

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
