# Copyright (c) 2021-2024, NVIDIA CORPORATION.
from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.lists.lists_column_view cimport lists_column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar


cdef extern from "cudf/lists/contains.hpp" namespace "cudf::lists" nogil:

    cpdef enum class duplicate_find_option(int32_t):
        FIND_FIRST
        FIND_LAST

    cdef unique_ptr[column] contains(
        const lists_column_view& lists,
        const scalar& search_key,
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] contains(
        const lists_column_view& lists,
        const column_view& search_keys,
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] contains_nulls(
        const lists_column_view& lists,
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] index_of(
        const lists_column_view& lists,
        const scalar& search_key,
        duplicate_find_option find_option,
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] index_of(
        const lists_column_view& lists,
        const column_view& search_keys,
        duplicate_find_option find_option,
    ) except +libcudf_exception_handler
