# Copyright (c) 2021, NVIDIA CORPORATION.

from cudf._lib.table cimport Table

from cudf._lib.cpp.copying cimport packed_columns

cdef class PackedColumns:
    cdef packed_columns c_obj
    cdef object column_names
    cdef object index_names

    @staticmethod
    cdef PackedColumns c_from_table(Table input_table, keep_index=*)

    cdef Table c_unpack(self)

    cdef const void* c_metadata_ptr(self) except *

    cdef size_t c_metadata_size(self) except *

    cdef void* c_gpu_data_ptr(self) except *

    cdef size_t c_gpu_data_size(self) except *
