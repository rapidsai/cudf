# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from pyarrow.includes.libarrow cimport RandomAccessFile
from cudf._lib.cpp.table.table cimport table


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
        VOID "cudf::experimental::io::io_type::VOID"
        USER_SINK "cudf::experimental::io::io_type::USER_SINK"

    ctypedef enum statistics_freq:
        STATISTICS_NONE = 0,
        STATISTICS_ROWGROUP = 1,
        STATISTICS_PAGE = 2,

    cdef cppclass table_metadata:
        table_metadata() except +

        vector[string] column_names
        map[string, string] user_data

    cdef cppclass table_metadata_with_nullability(table_metadata):
        table_metadata_with_nullability() except +

        vector[bool] nullability

    cdef cppclass table_with_metadata:
        unique_ptr[table] tbl
        table_metadata metadata

    cdef cppclass source_info:
        io_type type
        string filepath
        pair[const char*, size_t] buffer
        shared_ptr[RandomAccessFile] file

        source_info() except +
        source_info(const string filepath) except +
        source_info(const char* host_buffer, size_t size) except +
        source_info(const shared_ptr[RandomAccessFile] arrow_file) except +

    cdef cppclass sink_info:
        io_type type
        string filepath
        vector[char] * buffer
        data_sink * user_sink

        sink_info() except +
        sink_info(string file_path) except +
        sink_info(vector[char] * buffer) except +
        sink_info(data_sink * user_sink) except +


cdef extern from "cudf/io/data_sink.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass data_sink:
        pass
