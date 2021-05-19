# Copyright (c) 2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table_view cimport table_view

cdef extern from "cudf/lists/combine.hpp" namespace \
        "cudf::lists" nogil:
    cdef unique_ptr[column] concatenate_rows(
        const table_view input_table
    ) except +
