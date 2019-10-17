# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *

from libcpp.pair cimport pair
from libcpp.string cimport string

ctypedef uint32_t* bit_mask_t_ptr

cdef extern from "cudf/cudf.h" nogil:

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

    cdef gdf_error gdf_extract_datetime_weekday(
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

cdef extern from "cudf/transform.hpp" namespace "cudf" nogil:
    cdef gdf_column transform(
        const gdf_column& input,
        const string& ptx,
        gdf_dtype output_type,
        bool is_ptx
    ) except +

    cdef pair[bit_mask_t_ptr, gdf_size_type] nans_to_nulls(
        const gdf_column& input
    ) except +


cdef extern from "cudf/unary.hpp" namespace "cudf" nogil:

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
