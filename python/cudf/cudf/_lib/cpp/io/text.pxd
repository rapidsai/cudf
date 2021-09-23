# Copyright (c) 2020-2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.column.column cimport column


cdef extern from "cudf/io/text/data_chunk_source.hpp" \
        namespace "cudf::io::text" nogil:

    cdef cppclass data_chunk_source:
        data_chunk_source() except +

cdef extern from "cudf/io/text/data_chunk_source_factories.hpp" \
        namespace "cudf::io::text" nogil:

    unique_ptr[data_chunk_source] make_source(string data) except +
    unique_ptr[data_chunk_source] \
        make_source_from_file(string filename) except +


cdef extern from "cudf/io/text/multibyte_split.hpp" \
        namespace "cudf::io::text" nogil:

    unique_ptr[column] multibyte_split(data_chunk_source source,
                                       string delimiter) except +
