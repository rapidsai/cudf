# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.table.table_view cimport mutable_table_view, table_view
from pylibcudf.libcudf.types cimport size_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


cdef extern from "cudf/table/table.hpp" namespace "cudf" nogil:
    cdef cppclass table:
        table(
            const table&,
            cuda_stream_view stream,
            device_memory_resource* mr
        ) except +libcudf_exception_handler
        table(
            table_view,
            cuda_stream_view stream,
            device_memory_resource* mr
        ) except +libcudf_exception_handler
        size_type num_columns() except +libcudf_exception_handler
        size_type num_rows() except +libcudf_exception_handler
        table_view view() except +libcudf_exception_handler
        mutable_table_view mutable_view() except +libcudf_exception_handler
        vector[unique_ptr[column]] release() except +libcudf_exception_handler
