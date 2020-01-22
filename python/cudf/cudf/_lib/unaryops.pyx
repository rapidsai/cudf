# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import numba.cuda
import numba.numpy_support
import numpy as np

from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.string cimport string

import rmm
from rmm._lib.lib cimport cudaStream_t, c_free

from cudf.utils import cudautils

from cudf._libxx.column cimport Column
from cudf._lib.cudf cimport *
from cudf._lib.cudf import *
from cudf._lib.GDFError import GDFError


cimport cudf._lib.includes.unaryops as cpp_unaryops


_UNARY_OP = {
    'sin': cpp_unaryops.SIN,
    'cos': cpp_unaryops.COS,
    'tan': cpp_unaryops.TAN,
    'asin': cpp_unaryops.ARCSIN,
    'acos': cpp_unaryops.ARCCOS,
    'atan': cpp_unaryops.ARCTAN,
    'exp': cpp_unaryops.EXP,
    'log': cpp_unaryops.LOG,
    'sqrt': cpp_unaryops.SQRT,
    'ceil': cpp_unaryops.CEIL,
    'floor': cpp_unaryops.FLOOR,
    'abs': cpp_unaryops.ABS,
    'invert': cpp_unaryops.BIT_INVERT,
    'not': cpp_unaryops.NOT,
}


def apply_unary_op(Column incol, op):
    """
    Call Unary op on input column
    """
    from cudf.core.column import build_column

    cdef gdf_column* c_incol = column_view_from_column(incol)
    cdef gdf_column c_out_col
    cdef string cpp_str
    cdef gdf_dtype g_type
    cdef cpp_unaryops.unary_op c_op

    if callable(op):
        nb_type = numba.numpy_support.from_dtype(incol.dtype)
        type_signature = (nb_type,)
        compiled_op = cudautils.compile_udf(op, type_signature)
        cpp_str = compiled_op[0].encode('UTF-8')
        if compiled_op[1] not in dtypes:
            raise TypeError(
                "Result of window function has unsupported dtype {}"
                .format(op[1])
            )
        g_type = dtypes[compiled_op[1]]
        with nogil:
            c_out_col = cpp_unaryops.transform(
                c_incol[0],
                cpp_str,
                g_type,
                True
            )
    else:
        c_op = _UNARY_OP[op]
        with nogil:
            c_out_col = cpp_unaryops.unary_operation(
                c_incol[0],
                c_op
            )

    free_column(c_incol)

    return gdf_column_to_column(&c_out_col)


def apply_dt_extract_op(Column incol, Column outcol, op):
    """
    Call a datetime extraction op
    """
    cdef gdf_column* c_incol = column_view_from_column(incol)
    cdef gdf_column* c_outcol = column_view_from_column(outcol)

    cdef gdf_error result = GDF_CUDA_ERROR
    with nogil:
        if op == 'year':
            result = cpp_unaryops.gdf_extract_datetime_year(
                <gdf_column*>c_incol,
                <gdf_column*>c_outcol
            )
        elif op == 'month':
            result = cpp_unaryops.gdf_extract_datetime_month(
                <gdf_column*>c_incol,
                <gdf_column*>c_outcol
            )
        elif op == 'day':
            result = cpp_unaryops.gdf_extract_datetime_day(
                <gdf_column*>c_incol,
                <gdf_column*>c_outcol
            )
        elif op == 'weekday':
            result = cpp_unaryops.gdf_extract_datetime_weekday(
                <gdf_column*>c_incol,
                <gdf_column*>c_outcol
            )
        elif op == 'hour':
            result = cpp_unaryops.gdf_extract_datetime_hour(
                <gdf_column*>c_incol,
                <gdf_column*>c_outcol
            )
        elif op == 'minute':
            result = cpp_unaryops.gdf_extract_datetime_minute(
                <gdf_column*>c_incol,
                <gdf_column*>c_outcol
            )
        elif op == 'second':
            result = cpp_unaryops.gdf_extract_datetime_second(
                <gdf_column*>c_incol,
                <gdf_column*>c_outcol
            )

    free_column(c_incol)
    free_column(c_outcol)

    check_gdf_error(result)


def nans_to_nulls(Column py_col):
    from cudf.core.column import as_column
    from cudf.core.buffer import Buffer
    from cudf.utils.utils import mask_bitsize, calc_chunk_size

    cdef gdf_column* c_col = column_view_from_column(py_col)
    cdef pair[cpp_unaryops.bit_mask_t_ptr, size_type] result
    cdef cudaStream_t stream = <cudaStream_t><size_t>(0)

    with nogil:
        result = cpp_unaryops.nans_to_nulls(c_col[0])

    mask = None
    if result.second > 0:
        mask_ptr = int(<uintptr_t>result.first)
        mask_buf = rmm.DeviceBuffer(
            ptr=mask_ptr,
            size=calc_chunk_size(len(py_col), mask_bitsize)
        )
        mask = Buffer(mask_buf)
        c_free(result.first, stream)

    free_column(c_col)

    return mask


def is_null(Column col):
    cdef gdf_column* c_col = column_view_from_column(col)

    cdef gdf_column result = cpp_unaryops.is_null(c_col[0])
    free_column(c_col)

    return gdf_column_to_column(&result)


def is_not_null(Column col):
    cdef gdf_column* c_col = column_view_from_column(col)

    cdef gdf_column result = cpp_unaryops.is_not_null(c_col[0])
    free_column(c_col)

    return gdf_column_to_column(&result)
