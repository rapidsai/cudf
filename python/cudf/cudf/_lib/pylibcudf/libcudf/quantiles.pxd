# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view
from cudf._lib.pylibcudf.libcudf.types cimport (
    interpolation,
    null_order,
    order,
    order_info,
    sorted,
)


cdef extern from "cudf/quantiles.hpp" namespace "cudf" nogil:

    cdef unique_ptr[column] quantile (
        column_view input,
        vector[double] q,
        interpolation interp,
        column_view ordered_indices,
        bool exact,
    ) except +

    cdef unique_ptr[table] quantiles (
        table_view source_table,
        vector[double] q,
        interpolation interp,
        sorted is_input_sorted,
        vector[order] column_order,
        vector[null_order] null_precedence,
    ) except +
