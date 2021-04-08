# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.types cimport order, null_order
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.lists.lists_column_view cimport lists_column_view


cdef extern from "cudf/lists/sorting.hpp" namespace "cudf::lists" nogil:
    cdef unique_ptr[column] sort_lists(
        const lists_column_view source_column,
        order column_order,
        null_order null_precedence
    ) except +
