# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._lib.cpp.scalar.scalar cimport scalar

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.lists.lists_column_view cimport lists_column_view

from cudf._lib.cpp.column.column_view cimport column_view

cdef extern from "cudf/lists/contains.hpp" namespace "cudf::lists" nogil:
    cdef unique_ptr[column] contains(
        lists_column_view lists,
        scalar search_key,
    ) except +
