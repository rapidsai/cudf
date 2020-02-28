# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.lib cimport *
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.vector cimport vector


cdef extern from "cudf/io/types.hpp" \
        namespace "cudf::experimental::io" nogil:

    ctypedef enum quote_style:
        QUOTE_MINIMAL = 0,
        QUOTE_ALL,
        QUOTE_NONNUMERIC,
        QUOTE_NONE,

    ctypedef enum compression_type:
        NONE "cudf::experimental::io::compression_type::NONE"
        AUTO "cudf::experimental::io::compression_type::AUTO"
        SNAPPY "cudf::experimental::io::compression_type::SNAPPY"
        GZIP "cudf::experimental::io::compression_type::GZIP"
        BZIP2 "cudf::experimental::io::compression_type::BZIP2"
        BROTLI "cudf::experimental::io::compression_type::BROTLI"
        ZIP "cudf::experimental::io::compression_type::ZIP"
        XZ "cudf::experimental::io::compression_type::XZ"

    ctypedef enum io_type:
        FILEPATH "cudf::experimental::io::io_type::FILEPATH"
        HOST_BUFFER "cudf::experimental::io::io_type::HOST_BUFFER"
        ARROW_RANDOM_ACCESS_FILE \
            "cudf::experimental::io::io_type::ARROW_RANDOM_ACCESS_FILE"

    ctypedef enum statistics_freq:
        STATISTICS_NONE = 0,
        STATISTICS_ROWGROUP = 1,
        STATISTICS_PAGE = 2,

    cdef cppclass table_metadata:
        vector[string] column_names
        map[string, string] user_data

    cdef cppclass table_with_metadata:
        unique_ptr[table] tbl
        table_metadata metadata

    cdef cppclass source_info:
        io_type type
        string filepath
        pair[const char*, size_t] buffer

        source_info() except +
        source_info(const string filepath) except +
        source_info(const char* host_buffer, size_t size) except +

    cdef cppclass sink_info:
        io_type type
        string filepath

cdef extern from "<utility>" namespace "std" nogil:
    cdef source_info move(source_info)
    cdef table_with_metadata move(table_with_metadata)
