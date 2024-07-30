# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport int64_t, uint8_t
from libcpp cimport bool
from libcpp.functional cimport reference_wrapper
from libcpp.map cimport map
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector
from pylibcudf.libcudf.expressions cimport expression
from pylibcudf.libcudf.io.types cimport (
    compression_type,
    dictionary_policy,
    partition_info,
    sink_info,
    source_info,
    statistics_freq,
    table_input_metadata,
    table_with_metadata,
)
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport data_type, size_type


cdef extern from "cudf/io/parquet.hpp" namespace "cudf::io" nogil:
    cdef cppclass parquet_reader_options:
        parquet_reader_options() except +
        source_info get_source_info() except +
        vector[vector[size_type]] get_row_groups() except +
        const optional[reference_wrapper[expression]]& get_filter() except +
        data_type get_timestamp_type() except +
        bool is_enabled_use_pandas_metadata() except +
        bool is_enabled_arrow_schema() except +

        # setter

        void set_filter(expression &filter) except +
        void set_columns(vector[string] col_names) except +
        void set_num_rows(size_type val) except +
        void set_row_groups(vector[vector[size_type]] row_grp) except +
        void set_skip_rows(int64_t val) except +
        void enable_use_arrow_schema(bool val) except +
        void enable_use_pandas_metadata(bool val) except +
        void set_timestamp_type(data_type type) except +

        @staticmethod
        parquet_reader_options_builder builder(
            source_info src
        ) except +

    cdef cppclass parquet_reader_options_builder:
        parquet_reader_options_builder() except +
        parquet_reader_options_builder(
            source_info src
        ) except +
        parquet_reader_options_builder& columns(
            vector[string] col_names
        ) except +
        parquet_reader_options_builder& row_groups(
            vector[vector[size_type]] row_grp
        ) except +
        parquet_reader_options_builder& convert_strings_to_categories(
            bool val
        ) except +
        parquet_reader_options_builder& use_pandas_metadata(
            bool val
        ) except +
        parquet_reader_options_builder& use_arrow_schema(
            bool val
        ) except +
        parquet_reader_options_builder& timestamp_type(
            data_type type
        ) except +
        parquet_reader_options_builder& filter(
            const expression & f
        ) except +
        parquet_reader_options build() except +

    cdef table_with_metadata read_parquet(
        parquet_reader_options args) except +

    cdef cppclass parquet_writer_options_base:
        parquet_writer_options_base() except +
        sink_info get_sink_info() except +
        compression_type get_compression() except +
        statistics_freq get_stats_level() except +
        const optional[table_input_metadata]& get_metadata(
        ) except +
        size_t get_row_group_size_bytes() except +
        size_type get_row_group_size_rows() except +
        size_t get_max_page_size_bytes() except +
        size_type get_max_page_size_rows() except +
        size_t get_max_dictionary_size() except +
        bool is_enabled_write_arrow_schema() except +

        void set_metadata(
            table_input_metadata m
        ) except +
        void set_key_value_metadata(
            vector[map[string, string]] kvm
        ) except +
        void set_stats_level(
            statistics_freq sf
        ) except +
        void set_compression(
            compression_type compression
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
        void enable_write_arrow_schema(bool val) except +
        void set_dictionary_policy(dictionary_policy policy) except +

    cdef cppclass parquet_writer_options(parquet_writer_options_base):
        parquet_writer_options() except +
        table_view get_table() except +
        string get_column_chunks_file_paths() except +
        void set_partitions(
            vector[partition_info] partitions
        ) except +
        void set_column_chunks_file_paths(
            vector[string] column_chunks_file_paths
        ) except +

        @staticmethod
        parquet_writer_options_builder builder(
            sink_info sink_,
            table_view table_
        ) except +

    cdef cppclass parquet_writer_options_builder_base[BuilderT, OptionsT]:
        parquet_writer_options_builder_base() except +

        BuilderT& metadata(
            table_input_metadata m
        ) except +
        BuilderT& key_value_metadata(
            vector[map[string, string]] kvm
        ) except +
        BuilderT& stats_level(
            statistics_freq sf
        ) except +
        BuilderT& compression(
            compression_type compression
        ) except +
        BuilderT& int96_timestamps(
            bool enabled
        ) except +
        BuilderT& utc_timestamps(
            bool enabled
        ) except +
        BuilderT& write_arrow_schema(
            bool enabled
        ) except +
        BuilderT& row_group_size_bytes(
            size_t val
        ) except +
        BuilderT& row_group_size_rows(
            size_type val
        ) except +
        BuilderT& max_page_size_bytes(
            size_t val
        ) except +
        BuilderT& max_page_size_rows(
            size_type val
        ) except +
        BuilderT& max_dictionary_size(
            size_t val
        ) except +
        BuilderT& write_v2_headers(
            bool val
        ) except +
        BuilderT& dictionary_policy(
            dictionary_policy val
        ) except +
        OptionsT build() except +

    cdef cppclass parquet_writer_options_builder(
            parquet_writer_options_builder_base[parquet_writer_options_builder,
                                                parquet_writer_options]):
        parquet_writer_options_builder() except +
        parquet_writer_options_builder(
            sink_info sink_,
            table_view table_
        ) except +
        parquet_writer_options_builder& partitions(
            vector[partition_info] partitions
        ) except +
        parquet_writer_options_builder& column_chunks_file_paths(
            vector[string] column_chunks_file_paths
        ) except +

    cdef unique_ptr[vector[uint8_t]] write_parquet(
        parquet_writer_options args
    ) except +

    cdef cppclass chunked_parquet_writer_options(parquet_writer_options_base):
        chunked_parquet_writer_options() except +

        @staticmethod
        chunked_parquet_writer_options_builder builder(
            sink_info sink_,
        ) except +

    cdef cppclass chunked_parquet_writer_options_builder(
            parquet_writer_options_builder_base[chunked_parquet_writer_options_builder,
                                                chunked_parquet_writer_options]
            ):
        chunked_parquet_writer_options_builder() except +
        chunked_parquet_writer_options_builder(
            sink_info sink_,
        ) except +

    cdef cppclass parquet_chunked_writer:
        parquet_chunked_writer() except +
        parquet_chunked_writer(chunked_parquet_writer_options args) except +
        parquet_chunked_writer& write(
            table_view table_,
        ) except +
        parquet_chunked_writer& write(
            const table_view& table_,
            const vector[partition_info]& partitions,
        ) except +
        unique_ptr[vector[uint8_t]] close(
            vector[string] column_chunks_file_paths,
        ) except +

    cdef cppclass chunked_parquet_reader:
        chunked_parquet_reader() except +
        chunked_parquet_reader(
            size_t chunk_read_limit,
            const parquet_reader_options& options) except +
        chunked_parquet_reader(
            size_t chunk_read_limit,
            size_t pass_read_limit,
            const parquet_reader_options& options) except +
        bool has_next() except +
        table_with_metadata read_chunk() except +

    cdef unique_ptr[vector[uint8_t]] merge_row_group_metadata(
        const vector[unique_ptr[vector[uint8_t]]]& metadata_list
    ) except +
