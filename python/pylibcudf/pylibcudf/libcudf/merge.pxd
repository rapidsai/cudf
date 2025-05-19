# Copyright (c) 2020-2024, NVIDIA CORPORATION.
cimport pylibcudf.libcudf.types as libcudf_types
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view


cdef extern from "cudf/merge.hpp" namespace "cudf" nogil:
    cdef unique_ptr[table] merge (
        vector[table_view] tables_to_merge,
        vector[libcudf_types.size_type] key_cols,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence,
    ) except +libcudf_exception_handler
