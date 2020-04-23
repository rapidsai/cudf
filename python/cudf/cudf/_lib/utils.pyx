# Copyright (c) 2020, NVIDIA CORPORATION.

from cython.operator import dereference

from libcpp.vector cimport vector

from cudf._lib.column cimport Column
from cudf._lib.table cimport Table
from cudf._lib.cpp.column.column cimport column_view
from cudf._lib.cpp.table.table cimport table_view
from cudf._lib.move cimport move


cdef vector[column_view] make_column_views(object columns):
    cdef vector[column_view] views
    views.reserve(len(columns))
    for col in columns:
        views.push_back((<Column> col).view())
    return views


cdef vector[table_view] make_table_views(object tables):
    cdef vector[table_view] views
    views.reserve(len(tables))
    for tbl in tables:
        views.push_back((<Table> tbl).view())
    return views


cdef vector[table_view] make_table_data_views(object tables):
    cdef vector[table_view] views
    views.reserve(len(tables))
    for tbl in tables:
        views.push_back((<Table> tbl).data_view())
    return views


cdef class BufferArrayFromVector:
    @staticmethod
    cdef BufferArrayFromVector from_unique_ptr(
        unique_ptr[vector[uint8_t]] in_vec
    ):
        cdef BufferArrayFromVector buf = BufferArrayFromVector()
        buf.in_vec = move(in_vec)
        buf.length = dereference(buf.in_vec).size()
        return buf

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(uint8_t)

        self.shape[0] = self.length
        self.strides[0] = 1

        buffer.buf = dereference(self.in_vec).data()

        buffer.format = NULL  # byte
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.length * itemsize   # product(shape) * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass
