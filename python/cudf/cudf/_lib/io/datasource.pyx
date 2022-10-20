# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr
from pyarrow.includes.libarrow cimport CRandomAccessFile
from pyarrow.lib cimport NativeFile

from cudf._lib.cpp.io.types cimport arrow_io_source, datasource


cdef class Datasource:
    cdef datasource* get_datasource(self) nogil except *:
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
