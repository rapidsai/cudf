# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.pair cimport pair

from cudf._libxx.lib cimport *


cdef extern from "cudf/transform.hpp" namespace "cudf::experimental" nogil:
    cdef pair[unique_ptr[device_buffer], size_type] bools_to_mask (
        column_view input
    ) except +
