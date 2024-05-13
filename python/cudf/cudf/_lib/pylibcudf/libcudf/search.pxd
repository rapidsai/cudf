# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

cimport cudf._lib.cpp.types as libcudf_types
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table_view cimport table_view


cdef extern from "cudf/search.hpp" namespace "cudf" nogil:

    cdef unique_ptr[column] lower_bound(
        table_view haystack,
        table_view needles,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence,
    ) except +

    cdef unique_ptr[column] upper_bound(
        table_view haystack,
        table_view needles,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence,
    ) except +

    cdef unique_ptr[column] contains(
        column_view haystack,
        column_view needles,
    ) except +
