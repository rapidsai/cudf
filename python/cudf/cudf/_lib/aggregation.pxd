# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._lib.cpp.aggregation cimport aggregation


cdef unique_ptr[aggregation] make_aggregation(op, kwargs=*) except *

cdef class Aggregation:
    cdef unique_ptr[aggregation] c_obj
