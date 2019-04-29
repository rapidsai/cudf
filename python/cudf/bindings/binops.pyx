# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.binops cimport *
from cudf.bindings.GDFError import GDFError
from libcpp.vector cimport vector
from libc.stdlib cimport free

from librmm_cffi import librmm as rmm

_COMPILED_OPS = [
    'add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'eq', 'ne', 'lt', 'gt',
    'le', 'ge', 'and', 'or', 'xor'
]

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


cdef apply_jit_op(gdf_column* c_lhs, gdf_column* c_rhs, gdf_column* c_out, op):
    """
    Call JITified gdf binary ops.
    """

    cdef gdf_error result
    cdef gdf_binary_operator c_op = _BINARY_OP[op]
    with nogil:    
        result = gdf_binary_operation_v_v(
            <gdf_column*>c_out,
            <gdf_column*>c_lhs,
            <gdf_column*>c_rhs,
            c_op)

    cdef int nullct = c_out[0].null_count

    check_gdf_error(result)

    return nullct


cdef apply_mask_and(gdf_column* c_lhs, gdf_column* c_rhs, gdf_column* c_out):
    """

    """
    cdef gdf_error result

    with nogil:
        result = gdf_validity_and(
            <gdf_column*>c_lhs,
            <gdf_column*>c_rhs,
            <gdf_column*>c_out
        )

    check_gdf_error(result)

    cdef int nnz = 0
    if c_out.valid is not NULL:

        with nogil:
            nnz = gdf_count_nonzero_mask(
                c_out.valid,
                c_out.size,
                &nnz
            )

    return c_out.size - nnz


cdef apply_compiled_op(gdf_column* c_lhs, gdf_column* c_rhs, gdf_column* c_out, op):
    """
    Call compiled gdf binary ops.
    """

    cdef gdf_error result = GDF_CUDA_ERROR
    with nogil:
        if op == 'add':
            result = gdf_add_generic(
                <gdf_column*>c_lhs,
                <gdf_column*>c_rhs,
                <gdf_column*>c_out
            )
        elif op == 'sub':
            result = gdf_sub_generic(
                <gdf_column*>c_lhs,
                <gdf_column*>c_rhs,
                <gdf_column*>c_out
            )
        elif op == 'mul':
            result = gdf_mul_generic(
                <gdf_column*>c_lhs,
                <gdf_column*>c_rhs,
                <gdf_column*>c_out
            )
        elif op == 'div':
            result = gdf_div_generic(
                <gdf_column*>c_lhs,
                <gdf_column*>c_rhs,
                <gdf_column*>c_out
            )
        elif op == 'truediv':
            result = gdf_div_generic(
                <gdf_column*>c_lhs,
                <gdf_column*>c_rhs,
                <gdf_column*>c_out
            )
        elif op == 'floordiv':
            result = gdf_floordiv_generic(
                <gdf_column*>c_lhs,
                <gdf_column*>c_rhs,
                <gdf_column*>c_out
            )
        elif op == 'eq':
            result = gdf_eq_generic(
                <gdf_column*>c_lhs,
                <gdf_column*>c_rhs,
                <gdf_column*>c_out
            )
        elif op == 'ne':
            result = gdf_ne_generic(
                <gdf_column*>c_lhs,
                <gdf_column*>c_rhs,
                <gdf_column*>c_out
            )
        elif op == 'lt':
            result = gdf_lt_generic(
                <gdf_column*>c_lhs,
                <gdf_column*>c_rhs,
                <gdf_column*>c_out
            )
        elif op == 'gt':
            result = gdf_gt_generic(
                <gdf_column*>c_lhs,
                <gdf_column*>c_rhs,
                <gdf_column*>c_out
            )
        elif op == 'le':
            result = gdf_le_generic(
                <gdf_column*>c_lhs,
                <gdf_column*>c_rhs,
                <gdf_column*>c_out
            )
        elif op == 'ge':
            result = gdf_ge_generic(
                <gdf_column*>c_lhs,
                <gdf_column*>c_rhs,
                <gdf_column*>c_out
            )
        elif op == 'and':
            result = gdf_bitwise_and_generic(
                <gdf_column*>c_lhs,
                <gdf_column*>c_rhs,
                <gdf_column*>c_out
            )
        elif op == 'or':
            result = gdf_bitwise_or_generic(
                <gdf_column*>c_lhs,
                <gdf_column*>c_rhs,
                <gdf_column*>c_out
            )
        elif op == 'xor':
            result = gdf_bitwise_xor_generic(
                <gdf_column*>c_lhs,
                <gdf_column*>c_rhs,
                <gdf_column*>c_out
            )

    check_gdf_error(result)

    if c_out.valid is not NULL:
        return apply_mask_and(
            <gdf_column*>c_lhs,
            <gdf_column*>c_rhs,
            <gdf_column*>c_out
        )
    else:
        return 0


def apply_op(lhs, rhs, out, op):
    """
    Dispatches a binary op call to the appropriate libcudf function
    """
    check_gdf_compatibility(lhs)
    check_gdf_compatibility(rhs)
    check_gdf_compatibility(out)

    cdef gdf_column* c_lhs = column_view_from_column(lhs)
    cdef gdf_column* c_rhs = column_view_from_column(rhs)
    cdef gdf_column* c_out = column_view_from_column(out)

    cdef gdf_error result
    cdef gdf_binary_operator c_op = _BINARY_OP[op]

    if c_lhs.dtype == c_rhs.dtype and op in _COMPILED_OPS:
        try:
            nullct = apply_compiled_op(
                <gdf_column*>c_lhs,
                <gdf_column*>c_rhs,
                <gdf_column*>c_out,
                op
            )
        except GDFError as e:
            if e.errcode == b'GDF_UNSUPPORTED_DTYPE':
                nullct = apply_jit_op(
                    <gdf_column*>c_lhs,
                    <gdf_column*>c_rhs,
                    <gdf_column*>c_out,
                    op
                )
            else:
                raise e
    else:
        nullct = apply_jit_op(
            <gdf_column*>c_lhs,
            <gdf_column*>c_rhs,
            <gdf_column*>c_out,
            op
        )

    free(c_lhs)
    free(c_rhs)
    free(c_out)

    return nullct
