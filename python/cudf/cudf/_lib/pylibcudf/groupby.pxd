# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.aggregation cimport aggregation, groupby_aggregation
from cudf._lib.cpp.groupby cimport aggregation_request, groupby

from .column cimport Column


cdef class AggregationRequest:
    # The groupby APIs accept vectors of unique_ptrs to aggregation requests.
    # This ownership model means that if AggregationRequest owned the
    # corresponding C++ object, that object would have to be copied by e.g.
    # each groupby.aggregate call to avoid invalidating this object. Therefore,
    # this class instead stores only Python/Cython objects and constructs the
    # C++ object on the fly as requested.
    cdef Column values
    cdef list aggregations
    cdef aggregation_request to_libcudf(self) except *


cdef class GroupBy:
    cdef unique_ptr[groupby] c_obj
    cpdef tuple aggregate(self, list requests)
