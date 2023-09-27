# Copyright (c) 2023, NVIDIA CORPORATION.

from pyarrow cimport lib as pa

from cudf._lib.cpp.interop cimport column_metadata

from .scalar cimport Scalar
from .table cimport Table


cdef class ColumnMetadata:
    cdef public object name
    cdef public object children_meta
    cdef column_metadata to_libcudf(self)

cpdef Table from_arrow(
    pa.Table pyarrow_table,
)

cpdef pa.Table to_arrow(Table tbl, list metadata)
