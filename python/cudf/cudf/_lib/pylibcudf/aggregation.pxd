# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp cimport aggregation as cpp_aggregation
from cudf._lib.cpp.aggregation cimport (
    Kind as kind_t,
    aggregation,
    rolling_aggregation,
)


cdef class Aggregation:
    cdef aggregation *c_ptr
    cpdef kind_t kind(self)


cdef class SumAggregation(Aggregation):
    pass


cdef class RollingAggregation(SumAggregation):
    pass


cdef class GroupbyAggregation(Aggregation):
    pass


cdef class GroupbyScanAggregation(Aggregation):
    pass


cdef class ReduceAggregation(Aggregation):
    pass


cdef class ScanAggregation(Aggregation):
    pass
