# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib cimport pylibcudf
from cudf._lib.cpp.aggregation cimport (
    groupby_aggregation,
    groupby_scan_aggregation,
    reduce_aggregation,
    rolling_aggregation,
    scan_aggregation,
)


cdef class RollingAggregation:
    cdef unique_ptr[rolling_aggregation] c_obj

cdef class GroupbyAggregation:
    cdef pylibcudf.aggregation.Aggregation c_obj

cdef class ReduceAggregation:
    cdef unique_ptr[reduce_aggregation] c_obj

cdef class ScanAggregation:
    cdef unique_ptr[scan_aggregation] c_obj

cdef RollingAggregation make_rolling_aggregation(op, kwargs=*)
cdef GroupbyAggregation make_groupby_aggregation(op, kwargs=*)
cdef ReduceAggregation make_reduce_aggregation(op, kwargs=*)
cdef ScanAggregation make_scan_aggregation(op, kwargs=*)
