# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from .cudf_cpp cimport *
from .cudf_cpp import *

from librmm_cffi import librmm as rmm

from libc.stdlib cimport free


_BINARY_OP = {}
_BINARY_OP['add'] = GDF_ADD
_BINARY_OP['sub'] = GDF_SUB
_BINARY_OP['mul'] = GDF_MUL
_BINARY_OP['div'] = GDF_DIV
_BINARY_OP['truediv'] = GDF_TRUE_DIV
_BINARY_OP['floordiv'] = GDF_FLOOR_DIV
_BINARY_OP['mod'] = GDF_MOD
_BINARY_OP['pow'] = GDF_POW
_BINARY_OP['eq'] = GDF_EQUAL
_BINARY_OP['ne'] = GDF_NOT_EQUAL
_BINARY_OP['lt'] = GDF_LESS
_BINARY_OP['gt'] = GDF_GREATER
_BINARY_OP['le'] = GDF_LESS_EQUAL
_BINARY_OP['ge'] = GDF_GREATER_EQUAL
_BINARY_OP['and'] = GDF_BITWISE_AND
_BINARY_OP['or'] = GDF_BITWISE_OR
_BINARY_OP['xor'] = GDF_BITWISE_XOR


def apply_op(lhs, rhs, out, op):
    """
      Call JITified gdf binary ops.
    """

    check_gdf_compatibility(lhs)
    check_gdf_compatibility(rhs)
    check_gdf_compatibility(out)
    
    cdef gdf_column* c_lhs = column_view_from_column(lhs)
    cdef gdf_column* c_rhs = column_view_from_column(rhs)
    cdef gdf_column* c_out = column_view_from_column(out)

    cdef gdf_error result
    cdef gdf_binary_operator c_op = _BINARY_OP[op]
    with nogil:    
        result = gdf_binary_operation_v_v(
            <gdf_column*>c_out,
            <gdf_column*>c_lhs,
            <gdf_column*>c_rhs,
            c_op)

    cdef int nullct = c_out[0].null_count
    
    free(c_lhs)
    free(c_rhs)
    free(c_out)

    check_gdf_error(result)

    return nullct
