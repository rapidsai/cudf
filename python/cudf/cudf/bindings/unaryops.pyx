# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.cudf_cpp import *
from cudf.bindings.unaryops cimport *
from cudf.bindings.GDFError import GDFError
from libcpp.vector cimport vector

from librmm_cffi import librmm as rmm

from libcpp.string cimport string

import numpy as np

from cudf.utils import cudautils
import numba.cuda
import numba.numpy_support


_UNARY_OP = {
    'sin': SIN,
    'cos': COS,
    'tan': TAN,
    'asin': ARCSIN,
    'acos': ARCCOS,
    'atan': ARCTAN,
    'exp': EXP,
    'log': LOG,
    'sqrt': SQRT,
    'ceil': CEIL,
    'floor': FLOOR,
    'abs': ABS,
    'invert': BIT_INVERT,
    'not': NOT,
}


def apply_unary_op(incol, op):
    """
    Call Unary ops.
    """

    check_gdf_compatibility(incol)

    cdef gdf_column* c_incol = column_view_from_column(incol)

    cdef gdf_column c_out_col

    cdef unary_op c_op
    cdef string cpp_str
    cdef gdf_dtype g_type

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
            c_out_col = transform(c_incol[0], cpp_str, g_type, True)
    else:
        c_op = _UNARY_OP[op]
        with nogil:
            c_out_col = unary_operation(
                c_incol[0],
                c_op
            )

    free_column(c_incol)

    return gdf_column_to_column(&c_out_col)


def apply_dt_extract_op(incol, outcol, op):
    """
    Call a datetime extraction op
    """

    check_gdf_compatibility(incol)
    check_gdf_compatibility(outcol)

    cdef gdf_column* c_incol = column_view_from_column(incol)
    cdef gdf_column* c_outcol = column_view_from_column(outcol)

    cdef gdf_error result = GDF_CUDA_ERROR
    with nogil:
        if op == 'year':
            result = gdf_extract_datetime_year(
                <gdf_column*>c_incol,
                <gdf_column*>c_outcol
            )
        elif op == 'month':
            result = gdf_extract_datetime_month(
                <gdf_column*>c_incol,
                <gdf_column*>c_outcol
            )
        elif op == 'day':
            result = gdf_extract_datetime_day(
                <gdf_column*>c_incol,
                <gdf_column*>c_outcol
            )
        elif op == 'hour':
            result = gdf_extract_datetime_hour(
                <gdf_column*>c_incol,
                <gdf_column*>c_outcol
            )
        elif op == 'minute':
            result = gdf_extract_datetime_minute(
                <gdf_column*>c_incol,
                <gdf_column*>c_outcol
            )
        elif op == 'second':
            result = gdf_extract_datetime_second(
                <gdf_column*>c_incol,
                <gdf_column*>c_outcol
            )

    free_column(c_incol)
    free_column(c_outcol)

    check_gdf_error(result)
