# Copyright (c) 2021, NVIDIA CORPORATION.

from cudf._lib.table cimport Table

from cudf._lib.cpp.copying cimport packed_columns

cdef class PackedColumns:
    cdef packed_columns data
    cdef object column_names
    cdef object index_names

    @staticmethod
    cdef PackedColumns from_table(Table input_table, keep_index=*)

    cdef Table unpack(self)
