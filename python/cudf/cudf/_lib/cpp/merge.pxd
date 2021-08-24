# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

cimport cudf._lib.cpp.types as libcudf_types
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view


cdef extern from "cudf/merge.hpp" namespace "cudf" nogil:
    cdef unique_ptr[table] merge (
        vector[table_view] tables_to_merge,
        vector[libcudf_types.size_type] key_cols,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence,
    ) except +
