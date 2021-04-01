# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.lists.lists_column_view cimport lists_column_view
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.types cimport null_equality, nan_equality

cdef extern from "cudf/lists/drop_list_duplicates.hpp" \
        namespace "cudf::lists" nogil:
    cdef unique_ptr[column] drop_list_duplicates(
        const lists_column_view lists_column,
        null_equality nulls_equal,
        nan_equality nans_equal
    ) except +
