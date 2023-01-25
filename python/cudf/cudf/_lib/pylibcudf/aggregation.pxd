# Copyright (c) 2023, NVIDIA CORPORATION.

# from libcpp cimport bool as cbool
from libcpp.memory cimport unique_ptr

# from cudf._lib.cpp.column.column cimport column
# from cudf._lib.cpp.types cimport size_type
from cudf._lib.cpp.aggregation cimport aggregation, groupby_aggregation
from cudf._lib.cpp.groupby cimport aggregation_request, groupby

from .column_view cimport ColumnView

# from rmm._lib.device_buffer cimport DeviceBuffer


cdef class Aggregation:
    pass


cdef class GroupbyAggregation(Aggregation):
    # TODO: Maybe just store this by value instead of by pointer? But the
    # factories return a pointerso that's less natural and maybe
    # AggregationRequest should just use a unique_ptr for consistency.
    cdef unique_ptr[groupby_aggregation] c_obj
    cdef groupby_aggregation * get(self) nogil


cdef class AggregationRequest:
    # Requests are cheap and easy to work with, may as well store by value
    # instead of as a unique_ptr.
    # TODO: Probably need to change this since places want to pass by reference
    # and there's no easy way to write a nice getter so you end up having to
    # access the c_obj attribute directly.
    cdef aggregation_request c_obj
    cdef AggregationRequest copy(self)


cdef class GroupBy:
    cdef unique_ptr[groupby] c_obj
    cpdef aggregate(self, list requests)
    cdef groupby * get(self) nogil
