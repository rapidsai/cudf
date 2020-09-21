# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport table_view


cdef extern from "cudf/transpose.hpp" namespace "cudf" nogil:
    cdef unique_ptr[table] transpose(table_view input_table) except +
