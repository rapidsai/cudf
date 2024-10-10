# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp.vector cimport vector


cdef extern from "cudf/utilities/span.hpp" namespace "cudf" nogil:
    cdef cppclass host_span[T]:
        host_span() except +
        host_span(vector[T]) except +
    cdef cppclass device_span[T]:
        device_span()
        device_span(device_span other) except +
