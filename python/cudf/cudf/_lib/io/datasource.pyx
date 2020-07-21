# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from cudf._lib.cpp.io.types cimport datasource

cdef class Datasource:

    def __cinit__(self):
        print("__cinit__ in Datasource.pyx")

    cpdef init(self):
        print("init() in Datasource.pyx")
