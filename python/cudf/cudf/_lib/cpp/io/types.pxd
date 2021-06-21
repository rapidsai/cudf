# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from pyarrow.includes.libarrow cimport CRandomAccessFile
from cudf._lib.cpp.table.table cimport table


cdef extern from "cudf/io/types.hpp" \
        namespace "cudf::io" nogil:

    ctypedef enum quote_style:
        QUOTE_MINIMAL "cudf::io::quote_style::MINIMAL"
        QUOTE_ALL "cudf::io::quote_style::ALL"
        QUOTE_NONNUMERIC "cudf::io::quote_style::NONNUMERIC"
        QUOTE_NONE "cudf::io::quote_style::NONE"

    ctypedef enum compression_type:
        NONE "cudf::io::compression_type::NONE"
        AUTO "cudf::io::compression_type::AUTO"
        SNAPPY "cudf::io::compression_type::SNAPPY"
        GZIP "cudf::io::compression_type::GZIP"
        BZIP2 "cudf::io::compression_type::BZIP2"
        BROTLI "cudf::io::compression_type::BROTLI"
        ZIP "cudf::io::compression_type::ZIP"
        XZ "cudf::io::compression_type::XZ"

    ctypedef enum io_type:
        FILEPATH "cudf::io::io_type::FILEPATH"
        HOST_BUFFER "cudf::io::io_type::HOST_BUFFER"
        VOID "cudf::io::io_type::VOID"
        USER_IMPLEMENTED "cudf::io::io_type::USER_IMPLEMENTED"

    ctypedef enum statistics_freq:
        STATISTICS_NONE = 0,
        STATISTICS_ROWGROUP = 1,
        STATISTICS_PAGE = 2,

    cdef cppclass column_name_info:
        string name
        vector[column_name_info] children

    cdef cppclass table_metadata:
        table_metadata() except +

        vector[string] column_names
        map[string, string] user_data
        vector[column_name_info] schema_info

    cdef cppclass table_metadata_with_nullability(table_metadata):
        table_metadata_with_nullability() except +

        vector[bool] nullability

    cdef cppclass table_with_metadata:
        unique_ptr[table] tbl
        table_metadata metadata

    cdef cppclass host_buffer:
        const char* data
        size_t size

        host_buffer()
        host_buffer(const char* data, size_t size)

    cdef cppclass source_info:
        io_type type
        vector[string] filepaths
        vector[host_buffer] buffers
        vector[shared_ptr[CRandomAccessFile]] files

        source_info() except +
        source_info(const vector[string] &filepaths) except +
        source_info(const vector[host_buffer] &host_buffers) except +
        source_info(datasource *source) except +

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

cdef extern from "cudf/io/datasource.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass datasource:
        pass
