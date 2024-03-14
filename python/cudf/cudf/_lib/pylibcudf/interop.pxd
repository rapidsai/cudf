# Copyright (c) 2023, NVIDIA CORPORATION.

from cudf._lib.cpp.interop cimport column_metadata


cdef class ColumnMetadata:
    cdef public object name
    cdef public object children_meta
    cdef column_metadata to_libcudf(self)
