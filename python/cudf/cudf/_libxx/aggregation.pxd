# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._libxx.cpp.aggregation cimport aggregation


cdef class Aggregation:
    cdef unique_ptr[aggregation] c_obj
