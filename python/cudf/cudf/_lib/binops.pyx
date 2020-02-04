# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *
from cudf._lib.cudf import *
from cudf._lib.GDFError import GDFError
from libcpp.vector cimport vector
from libc.stdlib cimport free

from libcpp.string cimport string

import rmm

from cudf._lib.includes.binops cimport *


_BINARY_OP = {
    'add': GDF_ADD,
    'sub': GDF_SUB,
    'mul': GDF_MUL,
    'div': GDF_DIV,
    'truediv': GDF_TRUE_DIV,
    'floordiv': GDF_FLOOR_DIV,
    'mod': GDF_PYMOD,
    'pow': GDF_POW,
    'eq': GDF_EQUAL,
    'ne': GDF_NOT_EQUAL,
    'lt': GDF_LESS,
    'gt': GDF_GREATER,
    'le': GDF_LESS_EQUAL,
    'ge': GDF_GREATER_EQUAL,
    'and': GDF_BITWISE_AND,
    'or': GDF_BITWISE_OR,
    'xor': GDF_BITWISE_XOR,
    'l_and': GDF_LOGICAL_AND,
    'l_or': GDF_LOGICAL_OR,
}

cdef apply_op_v_v(gdf_column* c_lhs, gdf_column* c_rhs, gdf_column* c_out, op):
    """
    Call gdf binary ops between two columns.
    """

    cdef gdf_binary_operator c_op = _BINARY_OP[op]
    with nogil:
        binary_operation(
            <gdf_column*>c_out,
            <gdf_column*>c_lhs,
            <gdf_column*>c_rhs,
            c_op)

    cdef int nullct = c_out[0].null_count

    return nullct

cdef apply_op_v_s(gdf_column* c_lhs, gdf_scalar* c_rhs, gdf_column* c_out, op):
    """
    Call gdf binary ops between a column and a scalar.
    """

    cdef gdf_binary_operator c_op = _BINARY_OP[op]
    with nogil:
        binary_operation(
            <gdf_column*>c_out,
            <gdf_column*>c_lhs,
            <gdf_scalar*>c_rhs,
            c_op)

    cdef int nullct = c_out[0].null_count

    return nullct


cdef apply_op_s_v(gdf_scalar* c_lhs, gdf_column* c_rhs, gdf_column* c_out, op):
    """
    Call gdf binary ops between a scalar and a column.
    """

    cdef gdf_binary_operator c_op = _BINARY_OP[op]
    with nogil:
        binary_operation(
            <gdf_column*>c_out,
            <gdf_scalar*>c_lhs,
            <gdf_column*>c_rhs,
            c_op)

    cdef int nullct = c_out[0].null_count

    return nullct


def apply_op(lhs, rhs, out, op):
    """
    Dispatches a binary op call to the appropriate libcudf function:
    """
    check_gdf_compatibility(out)
    cdef gdf_column* c_lhs = NULL
    cdef gdf_column* c_rhs = NULL
    cdef gdf_scalar* c_scalar = NULL
    cdef gdf_column* c_out = column_view_from_column(out)

    if np.isscalar(lhs):
        check_gdf_compatibility(rhs)
        c_rhs = column_view_from_column(rhs)
        c_scalar = gdf_scalar_from_scalar(lhs)
        nullct = apply_op_s_v(
            <gdf_scalar*> c_scalar,
            <gdf_column*> c_rhs,
            <gdf_column*> c_out,
            op
        )
    elif lhs is None:
        check_gdf_compatibility(rhs)
        c_rhs = column_view_from_column(rhs)
        c_scalar = gdf_scalar_from_scalar(lhs, rhs.dtype)
        nullct = apply_op_s_v(
            <gdf_scalar*> c_scalar,
            <gdf_column*> c_rhs,
            <gdf_column*> c_out,
            op
        )

    elif np.isscalar(rhs):
        check_gdf_compatibility(lhs)
        c_lhs = column_view_from_column(lhs)
        c_scalar = gdf_scalar_from_scalar(rhs)
        nullct = apply_op_v_s(
            <gdf_column*> c_lhs,
            <gdf_scalar*> c_scalar,
            <gdf_column*> c_out,
            op
        )

    elif rhs is None:
        check_gdf_compatibility(lhs)
        c_lhs = column_view_from_column(lhs)
        c_scalar = gdf_scalar_from_scalar(rhs, lhs.dtype)
        nullct = apply_op_v_s(
            <gdf_column*> c_lhs,
            <gdf_scalar*> c_scalar,
            <gdf_column*> c_out,
            op
        )

    else:
        check_gdf_compatibility(lhs)
        check_gdf_compatibility(rhs)
        c_lhs = column_view_from_column(lhs)
        c_rhs = column_view_from_column(rhs)

        nullct = apply_op_v_v(
            <gdf_column*>c_lhs,
            <gdf_column*>c_rhs,
            <gdf_column*>c_out,
            op
        )

    free(c_scalar)
    free_column(c_lhs)
    free_column(c_rhs)
    free_column(c_out)

    return nullct


def apply_op_udf(lhs, rhs, udf_ptx, np_dtype):
    """
    Apply a user-defined binary operator (a UDF) defined in `udf_ptx` on
    the two input columns `lhs` and `rhs`. The output type of the UDF
    has to be specified in `np_dtype`, a numpy data type.
    Currently ONLY int32, int64, float32 and float64 are supported.
    """
    check_gdf_compatibility(lhs)
    check_gdf_compatibility(rhs)
    cdef gdf_column* c_lhs = column_view_from_column(lhs)
    cdef gdf_column* c_rhs = column_view_from_column(rhs)

    # get the gdf_type related to the input np type
    cdef gdf_dtype g_type = dtypes[np_dtype]

    cdef string cpp_str = udf_ptx.encode("UTF-8")

    cdef gdf_column c_out_col

    with nogil:
        c_out_col = binary_operation(
            <gdf_column>c_lhs[0],
            <gdf_column>c_rhs[0],
            cpp_str,
            g_type
        )

    free_column(c_lhs)
    free_column(c_rhs)

    return gdf_column_to_column(&c_out_col)
