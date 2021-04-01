# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._lib.cpp.aggregation cimport aggregation


cdef class Aggregation:
    cdef unique_ptr[aggregation] c_obj

cdef Aggregation make_aggregation(op, kwargs=*)
