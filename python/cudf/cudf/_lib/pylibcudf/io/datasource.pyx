# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr
from pyarrow.includes.libarrow cimport CRandomAccessFile
from pyarrow.lib cimport NativeFile

from cudf._lib.pylibcudf.libcudf.io.arrow_io_source cimport arrow_io_source
from cudf._lib.pylibcudf.libcudf.io.datasource cimport datasource


cdef class Datasource:
    cdef datasource* get_datasource(self) except * nogil:
        with gil:
            raise NotImplementedError("get_datasource() should not "
                                      + "be directly invoked here")

cdef class NativeFileDatasource(Datasource):

    def __cinit__(self, NativeFile native_file,):

        cdef shared_ptr[CRandomAccessFile] ra_src

        ra_src = native_file.get_random_access_file()
        self.c_datasource.reset(new arrow_io_source(ra_src))

    cdef datasource* get_datasource(self) nogil:
        return <datasource *> (self.c_datasource.get())
