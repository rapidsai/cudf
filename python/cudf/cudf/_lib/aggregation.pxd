# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.aggregation cimport aggregation, rolling_aggregation


cdef class Aggregation:
    cdef unique_ptr[aggregation] c_obj

cdef class RollingAggregation:
    cdef unique_ptr[rolling_aggregation] c_obj

cdef Aggregation make_aggregation(op, kwargs=*)
cdef RollingAggregation make_rolling_aggregation(op, kwargs=*)
