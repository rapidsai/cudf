# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr

from cudf._lib.exception_handler cimport cudf_exception_handler
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.lists.lists_column_view cimport (
    lists_column_view,
)
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport scalar


cdef extern from "cudf/lists/contains.hpp" namespace "cudf::lists" nogil:

    cpdef enum class duplicate_find_option(int32_t):
        FIND_FIRST "cudf::lists::duplicate_find_option::FIND_FIRST"
        FIND_LAST "cudf::lists::duplicate_find_option::FIND_LAST"

    cdef unique_ptr[column] contains(
        const lists_column_view& lists,
        const scalar& search_key,
    ) except +cudf_exception_handler

    cdef unique_ptr[column] contains(
        const lists_column_view& lists,
        const column_view& search_keys,
    ) except +cudf_exception_handler

    cdef unique_ptr[column] contains_nulls(
        lists_column_view lists,
    ) except +cudf_exception_handler

    cdef unique_ptr[column] index_of(
        lists_column_view lists,
        scalar search_key,
        # duplicate_find_option find_option,
    ) except +cudf_exception_handler

    cdef unique_ptr[column] index_of(
        lists_column_view lists,
        column_view search_keys,
        # duplicate_find_option find_option,
    ) except +cudf_exception_handler
