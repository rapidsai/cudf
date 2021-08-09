# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.io.types cimport (
    datasource,
    source_info,
    arrow_io_source,
)

from pyarrow.lib cimport NativeFile
from pyarrow.includes.libarrow cimport CRandomAccessFile


cdef class Datasource:
    cdef datasource* get_datasource(self) nogil except *:
        with gil:
            raise NotImplementedError("get_datasource() should not "
                                      + "be directly invoked here")

cdef class NativeFileDatasource(Datasource):

    def __cinit__(self, NativeFile native_file,):

        cdef shared_ptr[CRandomAccessFile] ra_src
        cdef arrow_io_source arrow_src

        ra_src = native_file.get_random_access_file()
        self.c_datasource = arrow_io_source(ra_src)

    cdef datasource* get_datasource(self) nogil:
        return <datasource *> &(self.c_datasource)
