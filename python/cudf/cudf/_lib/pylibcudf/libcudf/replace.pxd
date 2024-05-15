# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from cudf._lib.types import cudf_to_np_types, np_to_cudf_types

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport (
    column_view,
    mutable_column_view,
)
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport scalar


cdef extern from "cudf/replace.hpp" namespace "cudf" nogil:

    cpdef enum class replace_policy(bool):
        PRECEDING
        FOLLOWING

    cdef unique_ptr[column] replace_nulls(
        column_view source_column,
        column_view replacement_column) except +

    cdef unique_ptr[column] replace_nulls(
        column_view source_column,
        scalar replacement) except +

    cdef unique_ptr[column] replace_nulls(
        column_view source_column,
        replace_policy replace_policy) except +

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

    cdef unique_ptr[column] normalize_nans_and_zeros(
        column_view source_column) except +

    cdef void normalize_nans_and_zeros(
        mutable_column_view source_column) except +
