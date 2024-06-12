# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr

from cudf._lib.pylibcudf.libcudf.io.arrow_io_source cimport arrow_io_source
from cudf._lib.pylibcudf.libcudf.io.datasource cimport datasource


cdef class Datasource:
    cdef datasource* get_datasource(self) except * nogil


cdef class NativeFileDatasource(Datasource):
    cdef shared_ptr[arrow_io_source] c_datasource
    cdef datasource* get_datasource(self) nogil
