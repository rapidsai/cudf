# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.binops cimport *
from cudf.bindings.GDFError import GDFError
from cudf.dataframe.column import Column
from libcpp.vector cimport vector
from libc.stdlib cimport free

from libcpp.string cimport string

from librmm_cffi import librmm as rmm

_BINARY_OP = {
    'add'       : GDF_ADD,
    'sub'       : GDF_SUB,
    'mul'       : GDF_MUL,
    'div'       : GDF_DIV,
    'truediv'   : GDF_TRUE_DIV,
    'floordiv'  : GDF_FLOOR_DIV,
    'mod'       : GDF_PYMOD,
    'pow'       : GDF_POW,
    'eq'        : GDF_EQUAL,
    'ne'        : GDF_NOT_EQUAL,
    'lt'        : GDF_LESS,
    'gt'        : GDF_GREATER,
    'le'        : GDF_LESS_EQUAL,
    'ge'        : GDF_GREATER_EQUAL,
    'and'       : GDF_BITWISE_AND,
    'or'        : GDF_BITWISE_OR,
    'xor'       : GDF_BITWISE_XOR,
    'l_and'     : GDF_LOGICAL_AND,
    'l_or'      : GDF_LOGICAL_OR,
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

cdef apply_op_v_v_udf(gdf_column* c_lhs, gdf_column* c_rhs, gdf_column* c_out, ptx):
    """
    Call gdf binary ops between two columns using user-defined function (UDF) defined in "ptx".
    """

    cdef string cpp_str = ptx.encode('UTF-8')
    with nogil:
        binary_operation(
            <gdf_column*>c_out,
            <gdf_column*>c_lhs,
            <gdf_column*>c_rhs,
            cpp_str)

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

    free(c_lhs)
    free(c_rhs)
    free(c_scalar)
    free(c_out)

    return nullct
  
def apply_op_udf(lhs, rhs, out, ptx):
    """
    Dispatches a binary op call to the appropriate libcudf function:
    """
    check_gdf_compatibility(out)
    cdef gdf_column* c_lhs = NULL
    cdef gdf_column* c_rhs = NULL
    cdef gdf_column* c_out = column_view_from_column(out)

    check_gdf_compatibility(lhs)
    check_gdf_compatibility(rhs)
    c_lhs = column_view_from_column(lhs)
    c_rhs = column_view_from_column(rhs)

    nullct = apply_op_v_v_udf(
        <gdf_column*>c_lhs,
        <gdf_column*>c_rhs,
        <gdf_column*>c_out,
        ptx
    )

    free(c_lhs)
    free(c_rhs)
    free(c_out)

    return nullct
