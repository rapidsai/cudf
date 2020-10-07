# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._lib.cpp.io.types cimport datasource

cdef class Datasource:

    cdef datasource* get_datasource(self) nogil except *
