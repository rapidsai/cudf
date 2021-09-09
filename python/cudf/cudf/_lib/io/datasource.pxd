# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.io.types cimport arrow_io_source, datasource


cdef class Datasource:
    cdef datasource* get_datasource(self) nogil except *

cdef class NativeFileDatasource(Datasource):
    cdef arrow_io_source c_datasource
    cdef datasource* get_datasource(self) nogil
