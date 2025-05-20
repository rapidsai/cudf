# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.table.table_view cimport table_view
from rmm.pylibrmm.stream cimport Stream

cdef class Table:
    # List[pylibcudf.Column]
    cdef public list _columns

    cdef table_view view(self) nogil

    cpdef int num_columns(self)
    cpdef int num_rows(self)
    cpdef tuple shape(self)

    @staticmethod
    cdef Table from_libcudf(unique_ptr[table] libcudf_tbl, Stream stream=*)

    @staticmethod
    cdef Table from_table_view(const table_view& tv, Table owner)

    @staticmethod
    cdef Table from_table_view_of_arbitrary(const table_view& tv, object owner)

    cpdef list columns(self)
