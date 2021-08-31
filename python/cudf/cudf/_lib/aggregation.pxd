# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.aggregation cimport (
    aggregation,
    groupby_aggregation,
    groupby_scan_aggregation,
    rolling_aggregation,
)


cdef class Aggregation:
    cdef unique_ptr[aggregation] c_obj

cdef class RollingAggregation:
    cdef unique_ptr[rolling_aggregation] c_obj

cdef class GroupbyAggregation:
    cdef unique_ptr[groupby_aggregation] c_obj

cdef class GroupbyScanAggregation:
    cdef unique_ptr[groupby_scan_aggregation] c_obj

cdef Aggregation make_aggregation(op, kwargs=*)
cdef RollingAggregation make_rolling_aggregation(op, kwargs=*)
cdef GroupbyAggregation make_groupby_aggregation(op, kwargs=*)
cdef GroupbyScanAggregation make_groupby_scan_aggregation(op, kwargs=*)
