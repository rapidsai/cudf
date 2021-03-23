# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.lists.lists_column_view cimport lists_column_view

from cudf._lib.cpp.types cimport size_type

cdef extern from "cudf/lists/extract.hpp" namespace "cudf::lists" nogil:
    cdef unique_ptr[column] extract_list_element(
        const lists_column_view,
        size_type
    ) except +
