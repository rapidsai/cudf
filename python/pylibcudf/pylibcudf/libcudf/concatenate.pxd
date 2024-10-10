# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.libcudf.column.column cimport column, column_view
from pylibcudf.libcudf.table.table cimport table, table_view
from pylibcudf.libcudf.utilities.span cimport host_span

from rmm.librmm.device_buffer cimport device_buffer


cdef extern from "cudf/concatenate.hpp" namespace "cudf" nogil:
    # The versions of concatenate taking vectors don't exist in libcudf
    # C++, but passing a vector works because a host_span is implicitly
    # constructable from a vector. In case they are needed in the future,
    # host_span versions can be added, e.g:
    #
    # cdef unique_ptr[column] concatenate(host_span[column_view] columns) except +

    cdef unique_ptr[column] concatenate(const vector[column_view] columns) except +
    cdef unique_ptr[table] concatenate(const vector[table_view] tables) except +
