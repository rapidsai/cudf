# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.lists.lists_column_view cimport lists_column_view

cdef extern from "cudf/lists/count_elements.hpp" namespace "cudf::lists" nogil:
    cdef unique_ptr[column] count_elements(const lists_column_view) except +
