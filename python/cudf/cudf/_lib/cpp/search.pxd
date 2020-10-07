# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table_view cimport table_view
cimport cudf._lib.cpp.types as libcudf_types


cdef extern from "cudf/search.hpp" namespace "cudf" nogil:

    cdef unique_ptr[column] lower_bound(
        table_view t,
        table_view values,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence,
    ) except +

    cdef unique_ptr[column] upper_bound(
        table_view t,
        table_view values,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence,
    ) except +

    cdef unique_ptr[column] contains(
        column_view haystack,
        column_view needles,
    ) except +
