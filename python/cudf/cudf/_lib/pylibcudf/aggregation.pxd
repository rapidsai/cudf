# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.cpp cimport aggregation as cpp_aggregation
from cudf._lib.cpp.aggregation cimport Kind as kind_t, aggregation


cdef class Aggregation:
    cdef aggregation *c_ptr
    cpdef kind_t kind(self)
