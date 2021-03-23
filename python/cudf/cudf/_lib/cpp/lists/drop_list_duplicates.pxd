# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.lists.lists_column_view cimport lists_column_view


cdef extern from "cudf/lists/drop_list_duplicates.hpp" namespace "cudf::lists" nogil:
    cdef unique_ptr[column] drop_list_duplicates(
        const lists_column_view, 
        null_equality,
    ) except +
