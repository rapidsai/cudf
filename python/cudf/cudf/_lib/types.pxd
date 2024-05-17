# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp cimport bool

cimport cudf._lib.pylibcudf.libcudf.types as libcudf_types
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.lists.lists_column_view cimport (
    lists_column_view,
)

ctypedef bool underlying_type_t_order
ctypedef bool underlying_type_t_null_order
ctypedef bool underlying_type_t_sorted
ctypedef int32_t underlying_type_t_interpolation
ctypedef int32_t underlying_type_t_type_id
ctypedef bool underlying_type_t_null_policy

cdef dtype_from_column_view(column_view cv)

cdef libcudf_types.data_type dtype_to_data_type(dtype) except *
cpdef dtype_to_pylibcudf_type(dtype)
cdef bool is_decimal_type_id(libcudf_types.type_id tid) except *
