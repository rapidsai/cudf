# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport uint64_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.pylibcudf.libcudf.column.column cimport column


cdef extern from "cudf/io/text/byte_range_info.hpp" \
        namespace "cudf::io::text" nogil:

    cdef cppclass byte_range_info:
        byte_range_info() except +
        byte_range_info(size_t offset, size_t size) except +

cdef extern from "cudf/io/text/data_chunk_source.hpp" \
        namespace "cudf::io::text" nogil:

    cdef cppclass data_chunk_source:
        data_chunk_source() except +

cdef extern from "cudf/io/text/data_chunk_source_factories.hpp" \
        namespace "cudf::io::text" nogil:

    unique_ptr[data_chunk_source] make_source(string data) except +
    unique_ptr[data_chunk_source] \
        make_source_from_file(string filename) except +
    unique_ptr[data_chunk_source] \
        make_source_from_bgzip_file(string filename) except +
    unique_ptr[data_chunk_source] \
        make_source_from_bgzip_file(string filename,
                                    uint64_t virtual_begin,
                                    uint64_t virtual_end) except +


cdef extern from "cudf/io/text/multibyte_split.hpp" \
        namespace "cudf::io::text" nogil:

    cdef cppclass parse_options:
        byte_range_info byte_range
        bool strip_delimiters

        parse_options() except +

    unique_ptr[column] multibyte_split(data_chunk_source source,
                                       string delimiter,
                                       parse_options options) except +
