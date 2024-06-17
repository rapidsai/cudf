# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr

from pylibcudf.libcudf.io.datasource cimport datasource

cdef class Datasource:
    cdef datasource* get_datasource(self) except * nogil:
        with gil:
            raise NotImplementedError("get_datasource() should not "
                                      + "be directly invoked here")
