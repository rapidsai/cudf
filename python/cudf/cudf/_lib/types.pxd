# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t

from pylibcudf.libcudf.column.column_view cimport column_view

ctypedef int32_t underlying_type_t_type_id

cdef dtype_from_column_view(column_view cv)

cpdef dtype_to_pylibcudf_type(dtype)
