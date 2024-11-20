# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.expressions cimport expression
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport bitmask_type, data_type, size_type

from rmm.librmm.device_buffer cimport device_buffer


cdef extern from "cudf/transform.hpp" namespace "cudf" nogil:
    cdef pair[unique_ptr[device_buffer], size_type] bools_to_mask (
        column_view input
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] mask_to_bools (
        bitmask_type* bitmask, size_type begin_bit, size_type end_bit
    ) except +libcudf_exception_handler

    cdef pair[unique_ptr[device_buffer], size_type] nans_to_nulls(
        column_view input
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] compute_column(
        table_view table,
        expression expr
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] transform(
        column_view input,
        string unary_udf,
        data_type output_type,
        bool is_ptx
    ) except +libcudf_exception_handler

    cdef pair[unique_ptr[table], unique_ptr[column]] encode(
        table_view input
    ) except +libcudf_exception_handler

    cdef pair[unique_ptr[column], table_view] one_hot_encode(
        column_view input_column,
        column_view categories
    ) except +

    cdef unique_ptr[column] compute_column(
        const table_view table,
        const expression& expr
    ) except +libcudf_exception_handler
