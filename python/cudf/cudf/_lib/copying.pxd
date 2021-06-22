# Copyright (c) 2021, NVIDIA CORPORATION.

from cudf._lib.table cimport Table

from cudf._lib.cpp.copying cimport packed_columns

cdef class _CPackedColumns:
    cdef packed_columns c_obj
    cdef object column_names
    cdef object column_dtypes
    cdef object index_names

    @staticmethod
    cdef _CPackedColumns from_py_table(Table input_table, keep_index=*)
