# Copyright (c) 2021-2024, NVIDIA CORPORATION.
from pylibcudf.exception_handler import libcudf_exception_handler

from libcpp.vector cimport vector


cdef extern from "cudf/utilities/span.hpp" namespace "cudf" nogil:
    cdef cppclass host_span[T]:
        host_span() except +libcudf_exception_handler
        host_span(vector[T]) except +libcudf_exception_handler
