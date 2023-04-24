# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.aggregation cimport aggregation, groupby_aggregation

from .libcudf_types.column_view cimport ColumnView


cdef class Aggregation:
    pass


cdef class GroupbyAggregation(Aggregation):
    # TODO: Maybe just store this by value instead of by pointer? But the
    # factories return a pointerso that's less natural and maybe
    # AggregationRequest should just use a unique_ptr for consistency.
    cdef unique_ptr[groupby_aggregation] c_obj
    cdef groupby_aggregation * get(self) nogil
