# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr

from rmm._lib.device_buffer cimport device_buffer

from cudf._libxx.cpp.types cimport size_type
from cudf._libxx.cpp.column.column_view cimport column_view


cdef extern from "cudf/transform.hpp" namespace "cudf::experimental" nogil:
    cdef pair[unique_ptr[device_buffer], size_type] bools_to_mask (
        column_view input
    ) except +
