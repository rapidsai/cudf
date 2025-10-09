# Copyright (c) 2021-2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/lists/explode.hpp" namespace "cudf" nogil:
    cdef unique_ptr[table] explode_outer(
        const table_view,
        size_type explode_column_idx,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
