# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.table.table cimport table

from . cimport libcudf_classes


cdef class Table:
    # List[pylibcudf.Column]
    cdef object columns

    cdef libcudf_classes.TableView _underlying

    cpdef libcudf_classes.TableView get_underlying(self)

    @staticmethod
    cdef Table from_libcudf(unique_ptr[table] libcudf_tbl)
