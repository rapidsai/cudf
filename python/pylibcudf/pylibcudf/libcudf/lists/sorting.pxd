# Copyright (c) 2021-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.lists.lists_column_view cimport lists_column_view
from pylibcudf.libcudf.types cimport null_order, order


cdef extern from "cudf/lists/sorting.hpp" namespace "cudf::lists" nogil:
    cdef unique_ptr[column] sort_lists(
        const lists_column_view source_column,
        order column_order,
        null_order null_precedence
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] stable_sort_lists(
        const lists_column_view source_column,
        order column_order,
        null_order null_precedence
    ) except +libcudf_exception_handler
