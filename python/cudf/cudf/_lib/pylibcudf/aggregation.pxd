# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp cimport aggregation as cpp_aggregation
from cudf._lib.cpp.aggregation cimport (
    Kind as kind_t,
    aggregation,
    groupby_aggregation,
)
from cudf._lib.cpp.types cimport null_policy

ctypedef groupby_aggregation * gba_ptr

cdef class Aggregation:
    cdef unique_ptr[aggregation] c_ptr
    cpdef kind_t kind(self)
    cdef unique_ptr[groupby_aggregation] clone_as_groupby(self) except *

    @staticmethod
    cdef Aggregation from_libcudf(unique_ptr[aggregation] agg)
