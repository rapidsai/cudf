# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column, column_view
from cudf._lib.pylibcudf.libcudf.lists.lists_column_view cimport (
    lists_column_view,
)
from cudf._lib.pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/lists/extract.hpp" namespace "cudf::lists" nogil:
    cdef unique_ptr[column] extract_list_element(
        const lists_column_view,
        size_type
    ) except +
    cdef unique_ptr[column] extract_list_element(
        const lists_column_view,
        column_view
    ) except +
