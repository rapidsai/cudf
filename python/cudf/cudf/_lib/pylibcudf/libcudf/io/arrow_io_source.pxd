# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from pyarrow.includes.libarrow cimport CRandomAccessFile

cimport cudf._lib.pylibcudf.libcudf.io.datasource as cudf_io_datasource


cdef extern from "cudf/io/arrow_io_source.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass arrow_io_source(cudf_io_datasource.datasource):
        arrow_io_source(const string& arrow_uri) except +
        arrow_io_source(shared_ptr[CRandomAccessFile]) except +
