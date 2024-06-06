# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t, uint8_t
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from pyarrow.includes.libarrow cimport CRandomAccessFile

cimport cudf._lib.pylibcudf.libcudf.io.data_sink as cudf_io_data_sink
cimport cudf._lib.pylibcudf.libcudf.io.datasource as cudf_io_datasource
cimport cudf._lib.pylibcudf.libcudf.table.table_view as cudf_table_view
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/io/types.hpp" \
        namespace "cudf::io" nogil:

    cpdef enum class quote_style(int32_t):
        MINIMAL
        ALL
        NONNUMERIC
        NONE

    cpdef enum class compression_type(int32_t):
        NONE
        AUTO
        SNAPPY
        GZIP
        BZIP2
        BROTLI
        ZIP
        XZ
        ZLIB
        LZ4
        LZO
        ZSTD

    cpdef enum class io_type(int32_t):
        FILEPATH
        HOST_BUFFER
        VOID
        USER_IMPLEMENTED

    cpdef enum class statistics_freq(int32_t):
        STATISTICS_NONE = 0,
        STATISTICS_ROWGROUP = 1,
        STATISTICS_PAGE = 2,
        STATISTICS_COLUMN = 3,

    cpdef enum class dictionary_policy(int32_t):
        NEVER = 0,
        ADAPTIVE = 1,
        ALWAYS = 2,

    cdef extern from "cudf/io/types.hpp" namespace "cudf::io" nogil:
        cpdef enum class column_encoding(int32_t):
            USE_DEFAULT = -1
            DICTIONARY = 0
            PLAIN = 1
            DELTA_BINARY_PACKED = 2
            DELTA_LENGTH_BYTE_ARRAY =3
            DELTA_BYTE_ARRAY = 4
            BYTE_STREAM_SPLIT = 5
            DIRECT = 6
            DIRECT_V2 = 7
            DICTIONARY_V2 = 8

    cdef cppclass column_name_info:
        string name
        vector[column_name_info] children

    cdef cppclass table_metadata:
        table_metadata() except +

        vector[string] column_names
        map[string, string] user_data
        vector[unordered_map[string, string]] per_file_user_data
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
        column_in_metadata& set_output_as_binary(bool binary)
        column_in_metadata& set_type_length(int32_t type_length)
        column_in_metadata& set_skip_compression(bool skip)
        column_in_metadata& set_encoding(column_encoding enc)
        string get_name()

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
        source_info(cudf_io_datasource.datasource *source) except +
        source_info(const vector[cudf_io_datasource.datasource*] &datasources) except +

    cdef cppclass sink_info:
        io_type type
        const vector[string]& filepaths()
        const vector[vector[char] *]& buffers()
        const vector[cudf_io_data_sink.data_sink *]& user_sinks()

        sink_info() except +
        sink_info(string file_path) except +
        sink_info(vector[string] file_path) except +
        sink_info(vector[char] * buffer) except +
        sink_info(cudf_io_data_sink.data_sink * user_sink) except +
        sink_info(vector[cudf_io_data_sink.data_sink *] user_sink) except +
