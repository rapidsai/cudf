# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *


cdef extern from "cudf.h" nogil:

    cdef gdf_error gdf_extract_datetime_year(
        gdf_column *input,
        gdf_column *output
    ) except +

    cdef gdf_error gdf_extract_datetime_month(
        gdf_column *input,
        gdf_column *output
    ) except +

    cdef gdf_error gdf_extract_datetime_day(
        gdf_column *input,
        gdf_column *output
    ) except +

    cdef gdf_error gdf_extract_datetime_hour(
        gdf_column *input,
        gdf_column *output
    ) except +

    cdef gdf_error gdf_extract_datetime_minute(
        gdf_column *input,
        gdf_column *output
    ) except +

    cdef gdf_error gdf_extract_datetime_second(
        gdf_column *input,
        gdf_column *output
    ) except +


cdef extern from "unary.hpp" namespace "cudf" nogil:

    ctypedef enum unary_op:
        SIN,
        COS,
        TAN,
        ARCSIN,
        ARCCOS,
        ARCTAN,
        EXP,
        LOG,
        SQRT,
        CEIL,
        FLOOR,
        ABS,
        BIT_INVERT,
        NOT,
        INVALID_UNARY

    cdef gdf_column unary_operation(
        const gdf_column &input,
        unary_op op
    ) except +

    cdef gdf_column cast(
        const gdf_column &input,
        gdf_dtype out_type,
        gdf_dtype_extra_info out_info
    ) except +
