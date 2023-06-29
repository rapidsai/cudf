# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view


cdef class Table:
    # List[pylibcudf.Column]
    cdef object columns

    cdef unique_ptr[table_view] _view

    # See the corresponding pylibcudf.Column.view function for an
    # explanation of why we store a unique_ptr and return a raw pointer rather
    # than simply storing and returning table_view by value.
    cdef table_view* view(self)

    @staticmethod
    cdef Table from_libcudf(unique_ptr[table] libcudf_tbl)
