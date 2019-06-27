# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *


cdef extern from "cudf.h" nogil:

    cdef gdf_error gdf_extract_datetime_year(gdf_column *input, gdf_column *output) except +
    cdef gdf_error gdf_extract_datetime_month(gdf_column *input, gdf_column *output) except +
    cdef gdf_error gdf_extract_datetime_day(gdf_column *input, gdf_column *output) except +
    cdef gdf_error gdf_extract_datetime_hour(gdf_column *input, gdf_column *output) except +
    cdef gdf_error gdf_extract_datetime_minute(gdf_column *input, gdf_column *output) except +
    cdef gdf_error gdf_extract_datetime_second(gdf_column *input, gdf_column *output) except +


cdef extern from "unary.hpp" namespace "cudf" nogil:

    ctypedef enum unary_op:
        GDF_SIN,
        GDF_COS,
        GDF_TAN,
        GDF_ARCSIN,
        GDF_ARCCOS,
        GDF_ARCTAN,
        GDF_EXP,
        GDF_LOG,
        GDF_SQRT,
        GDF_CEIL,
        GDF_FLOOR,
        GDF_ABS,
        GDF_BIT_INVERT,
        GDF_NOT,
        GDF_INVALID_UNARY

    cdef gdf_column gdf_unaryop(gdf_column input, unary_op op) except +
