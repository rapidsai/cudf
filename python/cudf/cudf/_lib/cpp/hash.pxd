# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport uint32_t
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
cimport cudf._lib.cpp.types as libcudf_types


cdef extern from "cudf/hashing.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] hash "cudf::hash" (
        const table_view& input,
        const libcudf_types.hash_id& hash_function,
        const vector[uint32_t]& initial_hash,
        const uint32_t seed
    ) except +
