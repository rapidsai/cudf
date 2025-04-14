# Copyright (c) 2019-2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.libcudf.utilities.span cimport device_span

cdef extern from "cuda/functional" namespace "cuda::std":
    cdef cppclass byte:
        pass

cdef extern from "cudf/reshape.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] interleave_columns(
        table_view source_table
    ) except +libcudf_exception_handler
    cdef unique_ptr[table] tile(
        table_view source_table, size_type count
    ) except +libcudf_exception_handler

    void table_to_array(
        table_view input,
        device_span[byte] output,
        cudf.data_type output_dtype,
        rmm.cuda_stream_view stream
    ) except +libcudf_exception_handler
