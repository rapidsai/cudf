# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column, column_view
from pylibcudf.libcudf.table.table cimport table, table_view
from pylibcudf.libcudf.utilities.span cimport host_span

from rmm.librmm.device_buffer cimport device_buffer
from rmm.librmm.cuda_stream_view cimport cuda_stream_view


cdef extern from "cudf/concatenate.hpp" namespace "cudf" nogil:
    # The versions of concatenate taking vectors don't exist in libcudf
    # C++, but passing a vector works because a host_span is implicitly
    # constructable from a vector. In case they are needed in the future,
    # host_span versions can be added, e.g:
    #
    # cdef unique_ptr[column] concatenate(
    #    host_span[column_view] columns
    # ) except +libcudf_exception_handler

    cdef unique_ptr[column] concatenate(
        const vector[column_view] columns,
        cuda_stream_view stream
    ) except +libcudf_exception_handler
    cdef unique_ptr[table] concatenate(
        const vector[table_view] tables,
        cuda_stream_view stream
    ) except +libcudf_exception_handler
