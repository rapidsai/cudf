# Copyright (c) 2023, NVIDIA CORPORATION.

from . cimport libcudf_classes


cdef class Table:
    # List[pylibcudf.Column]
    cdef object columns

    cdef libcudf_classes.TableView _underlying

    cpdef libcudf_classes.TableView get_underlying(self)
