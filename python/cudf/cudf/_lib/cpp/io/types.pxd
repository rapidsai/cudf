# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector
from pyarrow.includes.libarrow cimport CRandomAccessFile

cimport cudf._lib.cpp.table.table_view as cudf_table_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.types cimport size_type


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
        ZLIB "cudf::io::compression_type::ZLIB"
        LZ4 "cudf::io::compression_type::LZ4"
        LZO "cudf::io::compression_type::LZO"
        ZSTD "cudf::io::compression_type::ZSTD"

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

    cdef cppclass table_with_metadata:
        unique_ptr[table] tbl
        table_metadata metadata

    cdef cppclass column_in_metadata:
        column_in_metadata& set_name(const string& name)
        column_in_metadata& set_nullability(bool nullable)
        column_in_metadata& set_list_column_as_map()
        column_in_metadata& set_int96_timestamps(bool req)
        column_in_metadata& set_decimal_precision(uint8_t precision)
        column_in_metadata& child(size_type i)

    cdef cppclass table_input_metadata:
        table_input_metadata() except +
        table_input_metadata(const cudf_table_view.table_view& table) except +

        vector[column_in_metadata] column_metadata

    cdef cppclass partition_info:
        size_type start_row
        size_type num_rows

        partition_info()
        partition_info(size_type start_row, size_type num_rows) except +

    cdef cppclass host_buffer:
        const char* data
        size_t size

        host_buffer()
        host_buffer(const char* data, size_t size)

    cdef cppclass source_info:
        io_type type
        const vector[string]& filepaths() except +
        const vector[host_buffer]& buffers() except +
        vector[shared_ptr[CRandomAccessFile]] files

        source_info() except +
        source_info(const vector[string] &filepaths) except +
        source_info(const vector[host_buffer] &host_buffers) except +
        source_info(datasource *source) except +

    cdef cppclass sink_info:
        io_type type
        const vector[string]& filepaths()
        const vector[vector[char] *]& buffers()
        const vector[data_sink *]& user_sinks()

        sink_info() except +
        sink_info(string file_path) except +
        sink_info(vector[string] file_path) except +
        sink_info(vector[char] * buffer) except +
        sink_info(data_sink * user_sink) except +
        sink_info(vector[data_sink *] user_sink) except +


cdef extern from "cudf/io/data_sink.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass data_sink:
        pass

cdef extern from "cudf/io/datasource.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass datasource:
        pass

    cdef cppclass arrow_io_source(datasource):
        arrow_io_source(string arrow_uri) except +
        arrow_io_source(shared_ptr[CRandomAccessFile]) except +
