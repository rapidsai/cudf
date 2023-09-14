# Copyright (c) 2023, NVIDIA CORPORATION.

from pyarrow.lib cimport Scalar as pa_Scalar, Table as pa_Table

from cudf._lib.cpp.interop cimport column_metadata

from .scalar cimport Scalar
from .table cimport Table


cdef class ColumnMetadata:
    cdef public object name
    cdef public object children_meta
    cdef column_metadata to_c_metadata(self)

cpdef Table from_arrow(
    pa_Table pyarrow_table,
)

cpdef Scalar from_arrow_scalar(
    pa_Scalar pyarrow_scalar,
)

cpdef pa_Table to_arrow(Table tbl, list metadata)

cpdef pa_Scalar to_arrow_scalar(Scalar slr, ColumnMetadata metadata)
