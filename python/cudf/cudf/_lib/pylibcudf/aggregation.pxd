# Copyright (c) 2023, NVIDIA CORPORATION.

# from libcpp cimport bool as cbool
from libcpp.memory cimport unique_ptr

# from cudf._lib.cpp.column.column cimport column
# from cudf._lib.cpp.types cimport size_type
from cudf._lib.cpp.aggregation cimport groupby_aggregation

# from libcpp.vector cimport vector

# from rmm._lib.device_buffer cimport DeviceBuffer


# from .column_view cimport ColumnView


cdef class Aggregation:
    pass


cdef class GroupbyAggregation(Aggregation):
    cdef unique_ptr[groupby_aggregation] c_obj
