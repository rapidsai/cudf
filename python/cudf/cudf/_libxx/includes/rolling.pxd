# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._libxx.types import np_to_cudf_types, cudf_to_np_types

from cudf._libxx.includes.types cimport size_type
from cudf._libxx.includes.column.column cimport column
from cudf._libxx.includes.column.column_view cimport column_view
from cudf._libxx.includes.aggregation cimport aggregation


cdef extern from "cudf/rolling.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[column] rolling_window(
        column_view source,
        column_view preceding_window,
        column_view following_window,
        size_type min_periods,
        unique_ptr[aggregation] agg) except +

    cdef unique_ptr[column] rolling_window(
        column_view source,
        size_type preceding_window,
        size_type following_window,
        size_type min_periods,
        unique_ptr[aggregation] agg) except +
