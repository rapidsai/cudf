# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.aggregation cimport (
    Kind as kind_t,
    aggregation,
    groupby_aggregation,
)
from cudf._lib.cpp.types cimport null_policy


cdef class Aggregation:
    cdef unique_ptr[aggregation] c_obj
    cpdef kind_t kind(self)
    cdef unique_ptr[groupby_aggregation] clone_underlying_as_groupby(self) except *

    @staticmethod
    cdef Aggregation from_libcudf(unique_ptr[aggregation] agg)
