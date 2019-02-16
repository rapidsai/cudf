# Copyright (c) 2018, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from .cudf_cpp cimport *
from .cudf_cpp import *


from librmm_cffi import librmm as rmm


from libc.stdint cimport uintptr_t
from libc.stdlib cimport free

from libcpp.map cimport map as cmap
from libcpp.string  cimport string as cstring

cdef cmap[cstring, gdf_binary_operator] _BINARY_OP
_BINARY_OP[b'add'] = GDF_ADD
_BINARY_OP[b'sub'] = GDF_SUB
_BINARY_OP[b'mul'] = GDF_MUL
_BINARY_OP[b'div'] = GDF_DIV
_BINARY_OP[b'truediv'] = GDF_TRUE_DIV
_BINARY_OP[b'floordiv'] = GDF_FLOOR_DIV
_BINARY_OP[b'mod'] = GDF_MOD
_BINARY_OP[b'pow'] = GDF_POW
_BINARY_OP[b'eq'] = GDF_EQUAL
_BINARY_OP[b'ne'] = GDF_NOT_EQUAL
_BINARY_OP[b'lt'] = GDF_LESS
_BINARY_OP[b'gt'] = GDF_GREATER
_BINARY_OP[b'le'] = GDF_LESS_EQUAL
_BINARY_OP[b'ge'] = GDF_GREATER_EQUAL


def apply_op(lhs, rhs, out, op):
    """
      Call JITified gdf binary ops.
    """

    oper = bytes(op, encoding="UTF-8")
    
    check_gdf_compatibility(lhs)
    check_gdf_compatibility(rhs)
    check_gdf_compatibility(out)
    
    cdef gdf_column* c_lhs = column_view_from_column(lhs)
    cdef gdf_column* c_rhs = column_view_from_column(rhs)
    cdef gdf_column* c_out = column_view_from_column(out)

    cdef gdf_error result
    cdef gdf_binary_operator c_op = _BINARY_OP[oper]
    with nogil:    
        result = gdf_binary_operation_v_v(
            <gdf_column*>c_out,
            <gdf_column*>c_lhs,
            <gdf_column*>c_rhs,
            c_op)

    free(c_lhs)
    free(c_rhs)
    free(c_out)

    check_gdf_error(result)
