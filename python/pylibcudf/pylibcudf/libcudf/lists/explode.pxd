# Copyright (c) 2021-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/lists/explode.hpp" namespace "cudf" nogil:
    cdef unique_ptr[table] explode_outer(
        const table_view,
        size_type explode_column_idx,
    ) except +libcudf_exception_handler
