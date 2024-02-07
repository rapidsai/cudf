# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib cimport pylibcudf
from cudf._lib.cpp.aggregation cimport rolling_aggregation


cdef class RollingAggregation:
    cdef unique_ptr[rolling_aggregation] c_obj

cdef class Aggregation:
    cdef pylibcudf.aggregation.Aggregation c_obj

cdef RollingAggregation make_rolling_aggregation(op, kwargs=*)
cdef Aggregation make_aggregation(op, kwargs=*)
