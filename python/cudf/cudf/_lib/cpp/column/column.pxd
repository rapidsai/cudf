# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp cimport bool

from rmm._lib.device_buffer cimport device_buffer
from cudf._lib.cpp.types cimport size_type, data_type
from cudf._lib.cpp.column.column_view cimport (
    column_view, mutable_column_view
)

cdef extern from "cudf/column/column.hpp" namespace "cudf" nogil:
    cdef cppclass column_contents "cudf::column::contents":
        unique_ptr[device_buffer] data
        unique_ptr[device_buffer] null_mask
        vector[unique_ptr[column]] children

    cdef cppclass column:
        column() except +
        column(const column& other) except +

        column(
            data_type dtype,
            size_type size,
            device_buffer&& data
        ) except +

        column(column_view view) except +

        size_type size() except +
        size_type null_count() except +
        bool has_nulls() except +
        data_type type() except +
        column_view view() except +
        mutable_column_view mutable_view() except +
        column_contents release() except +
