# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport table_view
cimport cudf._libxx.cpp.types as libcudf_types


cdef extern from "cudf/hashing.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] hash "cudf::hash" (
        const table_view& input,
        const vector[uint32_t]& initial_hash
    ) except +
