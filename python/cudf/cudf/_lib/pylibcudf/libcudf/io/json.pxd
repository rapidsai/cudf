# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

cimport cudf._lib.pylibcudf.libcudf.io.types as cudf_io_types
cimport cudf._lib.pylibcudf.libcudf.table.table_view as cudf_table_view
from cudf._lib.pylibcudf.libcudf.types cimport data_type, size_type


cdef extern from "cudf/io/json.hpp" \
        namespace "cudf::io" nogil:

    cdef struct schema_element:
        data_type type
        map[string, schema_element] child_types

    cdef cppclass json_reader_options:
        json_reader_options() except +
        cudf_io_types.source_info get_source() except +
        vector[string] get_dtypes() except +
        cudf_io_types.compression_type get_compression() except +
        size_type get_byte_range_offset() except +
        size_type get_byte_range_size() except +
        bool is_enabled_lines() except +
        bool is_enabled_mixed_types_as_string() except +
        bool is_enabled_prune_columns() except +
        bool is_enabled_dayfirst() except +
        bool is_enabled_experimental() except +

        # setter
        void set_dtypes(vector[data_type] types) except +
        void set_dtypes(map[string, schema_element] types) except +
        void set_compression(
            cudf_io_types.compression_type compression
        ) except +
        void set_byte_range_offset(size_type offset) except +
        void set_byte_range_size(size_type size) except +
        void enable_lines(bool val) except +
        void enable_mixed_types_as_string(bool val) except +
        void enable_prune_columns(bool val) except +
        void enable_dayfirst(bool val) except +
        void enable_experimental(bool val) except +
        void enable_keep_quotes(bool val) except +

        @staticmethod
        json_reader_options_builder builder(
            cudf_io_types.source_info src
        ) except +

    cdef cppclass json_reader_options_builder:
        json_reader_options_builder() except +
        json_reader_options_builder(
            cudf_io_types.source_info src
        ) except +
        json_reader_options_builder& dtypes(
            vector[string] types
        ) except +
        json_reader_options_builder& dtypes(
            vector[data_type] types
        ) except +
        json_reader_options_builder& dtypes(
            map[string, schema_element] types
        ) except +
        json_reader_options_builder& compression(
            cudf_io_types.compression_type compression
        ) except +
        json_reader_options_builder& byte_range_offset(
            size_type offset
        ) except +
        json_reader_options_builder& byte_range_size(
            size_type size
        ) except +
        json_reader_options_builder& lines(
            bool val
        ) except +
        json_reader_options_builder& mixed_types_as_string(
            bool val
        ) except +
        json_reader_options_builder& prune_columns(
            bool val
        ) except +
        json_reader_options_builder& dayfirst(
            bool val
        ) except +
        json_reader_options_builder& legacy(
            bool val
        ) except +
        json_reader_options_builder& keep_quotes(
            bool val
        ) except +

        json_reader_options build() except +

    cdef cudf_io_types.table_with_metadata read_json(
        json_reader_options &options) except +

    cdef cppclass json_writer_options:
        json_writer_options() except +
        cudf_io_types.sink_info get_sink() except +
        cudf_table_view.table_view get_table() except +
        string get_na_rep() except +
        bool is_enabled_include_nulls() except +
        bool is_enabled_lines() except +
        bool is_enabled_experimental() except +
        size_type get_rows_per_chunk() except +
        string get_true_value() except +
        string get_false_value() except +

        # setter
        void set_table(cudf_table_view.table_view tbl) except +
        void set_metadata(cudf_io_types.table_metadata meta) except +
        void set_na_rep(string val) except +
        void enable_include_nulls(bool val) except +
        void enable_lines(bool val) except +
        void set_rows_per_chunk(size_type val) except +
        void set_true_value(string val) except +
        void set_false_value(string val) except +

        @staticmethod
        json_writer_options_builder builder(
            cudf_io_types.sink_info sink,
            cudf_table_view.table_view tbl
        ) except +

    cdef cppclass json_writer_options_builder:
        json_writer_options_builder() except +
        json_writer_options_builder(
            cudf_io_types.source_info src,
            cudf_table_view.table_view tbl
        ) except +
        json_writer_options_builder& table(
            cudf_table_view.table_view tbl
        ) except +
        json_writer_options_builder& metadata(
            cudf_io_types.table_metadata meta
        ) except +
        json_writer_options_builder& na_rep(string val) except +
        json_writer_options_builder& include_nulls(bool val) except +
        json_writer_options_builder& lines(bool val) except +
        json_writer_options_builder& rows_per_chunk(size_type val) except +
        json_writer_options_builder& true_value(string val) except +
        json_writer_options_builder& false_value(string val) except +

        json_writer_options build() except +

    cdef cudf_io_types.table_with_metadata write_json(
        json_writer_options &options) except +
