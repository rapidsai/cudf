# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from pylibcudf.libcudf.io.datasource cimport datasource

__all__ = ["Datasource"]

cdef class Datasource:
    __hash__ = None
    cdef datasource* get_datasource(self) except * nogil:
        with gil:
            raise NotImplementedError("get_datasource() should not "
                                      + "be directly invoked here")
