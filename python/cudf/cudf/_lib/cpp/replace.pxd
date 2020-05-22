# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.types import np_to_cudf_types, cudf_to_np_types

from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport (
    column_view,
    mutable_column_view
)

cdef extern from "cudf/replace.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] replace_nulls(
        column_view source_column,
        column_view replacement_column) except +

    cdef unique_ptr[column] replace_nulls(
        column_view source_column,
        scalar replacement) except +

    cdef unique_ptr[column] find_and_replace_all(
        column_view source_column,
        column_view values_to_replace,
        column_view replacement_values) except +

    cdef unique_ptr[column] clamp(
        column_view source_column,
        scalar lo, scalar lo_replace,
        scalar hi, scalar hi_replace) except +

    cdef unique_ptr[column] clamp(
        column_view source_column,
        scalar lo, scalar hi) except +

cdef extern from "cudf/replace.hpp" namespace "cudf" nogil:
    cdef unique_ptr[column] normalize_nans_and_zeros(
        column_view source_column) except +

    cdef void normalize_nans_and_zeros(
        mutable_column_view source_column) except +
