# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from cudf._lib.cpp.column.column cimport column_view
from cudf._lib.cpp.table.table cimport table_view

cdef vector[column_view] make_column_views(object columns) except*
cdef vector[table_view] make_table_views(object tables) except*
cdef vector[table_view] make_table_data_views(object tables) except*

cdef class BufferArrayFromVector:
    cdef Py_ssize_t length
    cdef unique_ptr[vector[uint8_t]] in_vec

    # these two things declare part of the buffer interface
    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]

    @staticmethod
    cdef BufferArrayFromVector from_unique_ptr(
        unique_ptr[vector[uint8_t]] in_vec
    )
