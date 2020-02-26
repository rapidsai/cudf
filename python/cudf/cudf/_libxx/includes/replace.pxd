# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.lib import *
from cudf._libxx.lib cimport *

from cudf._libxx.includes.column.column cimport column
from cudf._libxx.includes.column.column_view cimport (
    column_view,
    mutable_column_view
)

cdef extern from "cudf/replace.hpp" namespace "cudf::experimental" nogil:
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
