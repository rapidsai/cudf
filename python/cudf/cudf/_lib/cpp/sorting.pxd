# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._lib.types import np_to_cudf_types, cudf_to_np_types

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view
cimport cudf._lib.cpp.types as libcudf_types

cdef extern from "cudf/sorting.hpp" namespace "cudf" nogil:
    ctypedef enum rank_method:
        FIRST "cudf::rank_method::FIRST"
        AVERAGE "cudf::rank_method::AVERAGE"
        MIN "cudf::rank_method::MIN"
        MAX "cudf::rank_method::MAX"
        DENSE "cudf::rank_method::DENSE"

cdef extern from "cudf/sorting.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] sorted_order(
        table_view source_table,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence) except +

    cdef unique_ptr[column] rank(
        column_view input_view,
        rank_method method,
        libcudf_types.order column_order,
        libcudf_types.null_policy null_handling,
        libcudf_types.null_order null_precedence,
        bool percentage) except +

    cdef bool is_sorted(
        const table_view& table,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence) except +
