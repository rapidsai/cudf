# Copyright (c) 2021-2023, NVIDIA CORPORATION.

from cudf._lib.cpp.contiguous_split cimport packed_columns


cdef class _CPackedColumns:
    cdef packed_columns c_obj
    cdef object column_names
    cdef object column_dtypes
    cdef object index_names
