# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t
from libcpp cimport bool
from libcpp.functional cimport reference_wrapper
from libcpp.map cimport map
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector

cimport cudf._lib.pylibcudf.libcudf.io.types as cudf_io_types
cimport cudf._lib.pylibcudf.libcudf.table.table_view as cudf_table_view
from cudf._lib.pylibcudf.libcudf.expressions cimport expression
from cudf._lib.pylibcudf.libcudf.types cimport data_type, size_type


cdef extern from "cudf/io/parquet.hpp" namespace "cudf::io" nogil:
    cdef cppclass parquet_reader_options:
        parquet_reader_options() except +
        cudf_io_types.source_info get_source_info() except +
        vector[vector[size_type]] get_row_groups() except +
        const optional[reference_wrapper[expression]]& get_filter() except +
        data_type get_timestamp_type() except +
        bool is_enabled_use_pandas_metadata() except +

        # setter

        void set_columns(vector[string] col_names) except +
        void set_row_groups(vector[vector[size_type]] row_grp) except +
        void enable_use_pandas_metadata(bool val) except +
        void set_timestamp_type(data_type type) except +

        @staticmethod
        parquet_reader_options_builder builder(
            cudf_io_types.source_info src
        ) except +

    cdef cppclass parquet_reader_options_builder:
        parquet_reader_options_builder() except +
        parquet_reader_options_builder(
            cudf_io_types.source_info src
        ) except +
        parquet_reader_options_builder& columns(
            vector[string] col_names
        ) except +
        parquet_reader_options_builder& row_groups(
            vector[vector[size_type]] row_grp
        ) except +
        parquet_reader_options_builder& use_pandas_metadata(
            bool val
        ) except +
        parquet_reader_options_builder& timestamp_type(
            data_type type
        ) except +
        parquet_reader_options_builder& filter(
            const expression & f
        ) except +
        parquet_reader_options build() except +

    cdef cudf_io_types.table_with_metadata read_parquet(
        parquet_reader_options args) except +

    cdef cppclass parquet_writer_options:
        parquet_writer_options() except +
        cudf_io_types.sink_info get_sink_info() except +
        cudf_io_types.compression_type get_compression() except +
        cudf_io_types.statistics_freq get_stats_level() except +
        cudf_table_view.table_view get_table() except +
        const optional[cudf_io_types.table_input_metadata]& get_metadata(
        ) except +
        string get_column_chunks_file_paths() except +
        size_t get_row_group_size_bytes() except +
        size_type get_row_group_size_rows() except +
        size_t get_max_page_size_bytes() except +
        size_type get_max_page_size_rows() except +
        size_t get_max_dictionary_size() except +

        void set_partitions(
            vector[cudf_io_types.partition_info] partitions
        ) except +
        void set_metadata(
            cudf_io_types.table_input_metadata m
        ) except +
        void set_key_value_metadata(
            vector[map[string, string]] kvm
        ) except +
        void set_stats_level(
            cudf_io_types.statistics_freq sf
        ) except +
        void set_compression(
            cudf_io_types.compression_type compression
        ) except +
        void set_column_chunks_file_paths(
            vector[string] column_chunks_file_paths
        ) except +
        void set_int96_timestamps(
            bool enabled
        ) except +
        void set_utc_timestamps(
            bool enabled
        ) except +
        void set_row_group_size_bytes(size_t val) except +
        void set_row_group_size_rows(size_type val) except +
        void set_max_page_size_bytes(size_t val) except +
        void set_max_page_size_rows(size_type val) except +
        void set_max_dictionary_size(size_t val) except +
        void enable_write_v2_headers(bool val) except +
        void set_dictionary_policy(cudf_io_types.dictionary_policy policy) except +

        @staticmethod
        parquet_writer_options_builder builder(
            cudf_io_types.sink_info sink_,
            cudf_table_view.table_view table_
        ) except +

    cdef cppclass parquet_writer_options_builder:

        parquet_writer_options_builder() except +
        parquet_writer_options_builder(
            cudf_io_types.sink_info sink_,
            cudf_table_view.table_view table_
        ) except +
        parquet_writer_options_builder& partitions(
            vector[cudf_io_types.partition_info] partitions
        ) except +
        parquet_writer_options_builder& metadata(
            cudf_io_types.table_input_metadata m
        ) except +
        parquet_writer_options_builder& key_value_metadata(
            vector[map[string, string]] kvm
        ) except +
        parquet_writer_options_builder& stats_level(
            cudf_io_types.statistics_freq sf
        ) except +
        parquet_writer_options_builder& compression(
            cudf_io_types.compression_type compression
        ) except +
        parquet_writer_options_builder& column_chunks_file_paths(
            vector[string] column_chunks_file_paths
        ) except +
        parquet_writer_options_builder& int96_timestamps(
            bool enabled
        ) except +
        parquet_writer_options_builder& utc_timestamps(
            bool enabled
        ) except +
        parquet_writer_options_builder& row_group_size_bytes(
            size_t val
        ) except +
        parquet_writer_options_builder& row_group_size_rows(
            size_type val
        ) except +
        parquet_writer_options_builder& max_page_size_bytes(
            size_t val
        ) except +
        parquet_writer_options_builder& max_page_size_rows(
            size_type val
        ) except +
        parquet_writer_options_builder& max_dictionary_size(
            size_t val
        ) except +
        parquet_writer_options_builder& write_v2_headers(
            bool val
        ) except +
        parquet_writer_options_builder& dictionary_policy(
            cudf_io_types.dictionary_policy val
        ) except +

        parquet_writer_options build() except +

    cdef unique_ptr[vector[uint8_t]] write_parquet(
        parquet_writer_options args
    ) except +

    cdef cppclass chunked_parquet_writer_options:
        chunked_parquet_writer_options() except +
        cudf_io_types.sink_info get_sink() except +
        cudf_io_types.compression_type get_compression() except +
        cudf_io_types.statistics_freq get_stats_level() except +
        const optional[cudf_io_types.table_input_metadata]& get_metadata(
        ) except +
        size_t get_row_group_size_bytes() except +
        size_type get_row_group_size_rows() except +
        size_t get_max_page_size_bytes() except +
        size_type get_max_page_size_rows() except +
        size_t get_max_dictionary_size() except +

        void set_metadata(
            cudf_io_types.table_input_metadata m
        ) except +
        void set_key_value_metadata(
            vector[map[string, string]] kvm
        ) except +
        void set_stats_level(
            cudf_io_types.statistics_freq sf
        ) except +
        void set_compression(
            cudf_io_types.compression_type compression
        ) except +
        void set_int96_timestamps(
            bool enabled
        ) except +
        void set_utc_timestamps(
            bool enabled
        ) except +
        void set_row_group_size_bytes(size_t val) except +
        void set_row_group_size_rows(size_type val) except +
        void set_max_page_size_bytes(size_t val) except +
        void set_max_page_size_rows(size_type val) except +
        void set_max_dictionary_size(size_t val) except +
        void enable_write_v2_headers(bool val) except +
        void set_dictionary_policy(cudf_io_types.dictionary_policy policy) except +

        @staticmethod
        chunked_parquet_writer_options_builder builder(
            cudf_io_types.sink_info sink_,
        ) except +

    cdef cppclass chunked_parquet_writer_options_builder:
        chunked_parquet_writer_options_builder() except +
        chunked_parquet_writer_options_builder(
            cudf_io_types.sink_info sink_,
        ) except +
        chunked_parquet_writer_options_builder& metadata(
            cudf_io_types.table_input_metadata m
        ) except +
        chunked_parquet_writer_options_builder& key_value_metadata(
            vector[map[string, string]] kvm
        ) except +
        chunked_parquet_writer_options_builder& stats_level(
            cudf_io_types.statistics_freq sf
        ) except +
        chunked_parquet_writer_options_builder& compression(
            cudf_io_types.compression_type compression
        ) except +
        chunked_parquet_writer_options_builder& int96_timestamps(
            bool enabled
        ) except +
        chunked_parquet_writer_options_builder& utc_timestamps(
            bool enabled
        ) except +
        chunked_parquet_writer_options_builder& row_group_size_bytes(
            size_t val
        ) except +
        chunked_parquet_writer_options_builder& row_group_size_rows(
            size_type val
        ) except +
        chunked_parquet_writer_options_builder& max_page_size_bytes(
            size_t val
        ) except +
        chunked_parquet_writer_options_builder& max_page_size_rows(
            size_type val
        ) except +
        chunked_parquet_writer_options_builder& max_dictionary_size(
            size_t val
        ) except +
        parquet_writer_options_builder& write_v2_headers(
            bool val
        ) except +
        parquet_writer_options_builder& dictionary_policy(
            cudf_io_types.dictionary_policy val
        ) except +

        chunked_parquet_writer_options build() except +

    cdef cppclass parquet_chunked_writer:
        parquet_chunked_writer() except +
        parquet_chunked_writer(chunked_parquet_writer_options args) except +
        parquet_chunked_writer& write(
            cudf_table_view.table_view table_,
        ) except +
        parquet_chunked_writer& write(
            const cudf_table_view.table_view& table_,
            const vector[cudf_io_types.partition_info]& partitions,
        ) except +
        unique_ptr[vector[uint8_t]] close(
            vector[string] column_chunks_file_paths,
        ) except +

    cdef unique_ptr[vector[uint8_t]] merge_row_group_metadata(
        const vector[unique_ptr[vector[uint8_t]]]& metadata_list
    ) except +
