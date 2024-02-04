# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib cimport pylibcudf


cdef class Aggregation:
    cdef pylibcudf.aggregation.Aggregation c_obj

cdef Aggregation make_aggregation(op, kwargs=*)
