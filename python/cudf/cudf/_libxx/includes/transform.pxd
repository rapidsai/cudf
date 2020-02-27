# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.pair cimport pair

from cudf._libxx.lib cimport *


cdef extern from "cudf/transform.hpp" namespace "cudf::experimental" nogil:
    cdef pair[unique_ptr[device_buffer], size_type] bools_to_mask (
        column_view input
    ) except +

    cdef pair[unique_ptr[device_buffer], size_type] nans_to_nulls(
        column_view input
    ) except +
