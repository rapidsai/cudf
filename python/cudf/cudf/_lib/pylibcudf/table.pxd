# Copyright (c) 2023, NVIDIA CORPORATION.

from . cimport libcudf_types


cdef class Table:
    # List[pylibcudf.Column]
    cdef object columns

    cdef libcudf_types.TableView _underlying

    cpdef libcudf_types.TableView get_underlying(self)
