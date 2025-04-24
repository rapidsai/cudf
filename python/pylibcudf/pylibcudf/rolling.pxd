# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from pylibcudf.libcudf.rolling cimport (
    bounded_closed, bounded_open, current_row, rolling_request, unbounded
)
from pylibcudf.libcudf.types cimport null_order, order, size_type

from .aggregation cimport Aggregation
from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table
from .types cimport DataType


ctypedef fused WindowType:
    Column
    size_type

cdef class Unbounded:
    cdef unique_ptr[unbounded] c_obj

cdef class CurrentRow:
    cdef unique_ptr[current_row] c_obj

cdef class BoundedClosed:
    cdef readonly Scalar delta
    cdef unique_ptr[bounded_closed] c_obj

cdef class BoundedOpen:
    cdef readonly Scalar delta
    cdef unique_ptr[bounded_open] c_obj

ctypedef fused PrecedingRangeWindowType:
    BoundedClosed
    BoundedOpen
    CurrentRow
    Unbounded

ctypedef fused FollowingRangeWindowType:
    BoundedClosed
    BoundedOpen
    CurrentRow
    Unbounded

cdef class RollingRequest:
    cdef Column values
    cdef Aggregation aggregation
    cdef size_type min_periods

    cdef rolling_request view(self) except *

cpdef Table grouped_range_rolling_window(
    Table group_keys,
    Column orderby,
    order order,
    null_order null_order,
    PrecedingRangeWindowType preceding,
    FollowingRangeWindowType following,
    list requests,
)

cpdef Column rolling_window(
    Column source,
    WindowType preceding_window,
    WindowType following_window,
    size_type min_periods,
    Aggregation agg,
)

cpdef bool is_valid_rolling_aggregation(DataType source, Aggregation agg)
