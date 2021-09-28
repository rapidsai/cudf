# Copyright (c) 2021, NVIDIA CORPORATION.

from cudf._lib.cpp.copying cimport packed_columns


cdef class _CPackedColumns:
    cdef packed_columns c_obj
    cdef object column_names
    cdef object column_dtypes
    cdef object index_names
