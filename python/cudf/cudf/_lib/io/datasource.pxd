# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.io.types cimport datasource, arrow_io_source


cdef class Datasource:
    cdef datasource* get_datasource(self) nogil except *

cdef class NativeFileDatasource(Datasource):
    cdef arrow_io_source c_datasource    
    cdef datasource* get_datasource(self) nogil
