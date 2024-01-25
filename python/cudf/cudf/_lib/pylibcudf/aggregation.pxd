# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.cpp.aggregation cimport Kind as kind_t, aggregation


cdef class Aggregation:
    cdef aggregation *c_ptr
    cpdef kind_t kind(self)


cdef class RollingAggregation(Aggregation):
    pass


cdef class GroupbyAggregation(Aggregation):
    pass


cdef class GroupbyScanAggregation(Aggregation):
    pass


cdef class ReduceAggregation(Aggregation):
    pass


cdef class ScanAggregation(Aggregation):
    pass
