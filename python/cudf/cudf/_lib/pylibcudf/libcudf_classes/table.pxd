# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.table.table cimport table


cdef class Table:
    cdef unique_ptr[table] c_obj
    cdef table * get(self) noexcept

    @staticmethod
    cdef from_table(unique_ptr[table] tbl)
