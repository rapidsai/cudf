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
from libc.stdlib cimport malloc, free

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

# TODO Not sure where to put this
def _is_single_value(val):
    from pandas.core.dtypes.common import is_datetime_or_timedelta_dtype
    return (
            isinstance(val, str)
            or isinstance(val, numbers.Number)
            or is_datetime_or_timedelta_dtype(val)
            or isinstance(val, pd.Timestamp)
            or isinstance(val, pd.Categorical)
            )


def apply_op(lhs, rhs, out, op):
    """
    Dispatches a binary op call to the appropriate libcudf function
    """
    check_gdf_compatibility(out)
    cdef gdf_column* c_out = column_view_from_column(out)

    cdef gdf_error result
    cdef gdf_binary_operator c_op = _BINARY_OP[op]
    cdef gdf_scalar* s
    cdef gdf_column* c_col
    cdef gdf_column* c_lhs
    cdef gdf_column* c_rhs

    # Simultaneously track whether we have any scalars, and which one
    # TODO is this the cleanest way?
    left = None

    # Check if either lhs or rhs are scalars
    # TODO do we need to check if both are scalars?
    if not isinstance(lhs, Column):
        s = gdf_scalar_from_scalar(lhs)
        left = True
        c_col = column_view_from_column(rhs)
    else:
        check_gdf_compatibility(lhs)
        c_lhs = column_view_from_column(lhs)

    if not isinstance(rhs, Column):
        s = gdf_scalar_from_scalar(rhs)
        left = False
        c_col = column_view_from_column(lhs)
    else:
        check_gdf_compatibility(rhs)
        c_rhs = column_view_from_column(rhs)

    # Careful, because None is a sentinel value here, `if left:` doesn't work
    if left is not None:

        nullct = apply_scalar_op(
                 s,
                 c_col,
                 c_out,
                 c_op,
                 left)
        free(c_out)
        free(s)
        free(c_col)
        return nullct

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

cdef apply_scalar_op(gdf_scalar *s, gdf_column *col, gdf_column *out, 
                     gdf_binary_operator op, left):
    cdef gdf_error result

    if left:
        result = gdf_binary_operation_s_v(
                <gdf_column*>out,
                <gdf_scalar*>s,
                <gdf_column*>col,
                <gdf_binary_operator>op)
    else:
        result = gdf_binary_operation_v_s(
                <gdf_column*>out,
                <gdf_column*>col,
                <gdf_scalar*>s,
                <gdf_binary_operator>op)

    check_gdf_error(result)
    cdef int nullct = out[0].null_count
    return nullct


# // def apply_scalar_eq(lhs, s_dtype, rhs, out):
# //     check_gdf_compatibility(rhs)
# //     check_gdf_compatibility(out)

# //     cdef gdf_scalar* s_lhs = <gdf_scalar*>malloc(sizeof(gdf_scalar))
# //     if s_lhs == NULL:
# //         # TODO what do?

# //     s_lhs.gdf_data = lhs
# //     s_lhs.gdf_dtype = s_dtype
# //     cdef gdf_column* c_rhs = column_view_from_column(rhs)
# //     cdef gdf_column* out = column_view_from_column(out)

# //     if s_lhs.dtype == c_rhs.dtype:
# //         cdef gdf_binary_operator op = _BINARY_OP['eq']
# //         cdef gdf_error result
# //         result = gdf_binary_operation_s_v(
# //             <gdf_column*>out,
# //             <gdf_scalar*>s_lhs,
# //             <gdf_column*>c_rhs,

# //         )

# //     check_gdf_error(result)
# //     free(s_lhs)
# //     free(c_rhs) # TODO do i need to free this?
# //     free(out) # TODO do i need to free this?
