# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.pair cimport pair

from cudf._libxx.lib cimport *

from rmm._lib.device_buffer cimport device_buffer
from cudf._libxx.includes.column.column_view cimport column_view


cdef extern from "cudf/transform.hpp" namespace "cudf::experimental" nogil:
    cdef pair[unique_ptr[device_buffer], size_type] bools_to_mask (
        column_view input
    ) except +
