# Copyright (c) 2021-2024, NVIDIA CORPORATION.

# from pylibcudf.libcudf.contiguous_split cimport packed_columns
cimport pylibcudf as plc

cdef class _CPackedColumns:
    cdef plc.contiguous_split.PackedColumns c_obj
    cdef object column_names
    cdef object column_dtypes
    cdef object index_names
