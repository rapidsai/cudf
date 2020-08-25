# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, unique_ptr
from libc.stdint cimport uint8_t

from cudf._lib.cpp.types cimport data_type, size_type
cimport cudf._lib.cpp.io.types as cudf_io_types
cimport cudf._lib.cpp.table.table_view as cudf_table_view

cdef extern from "cudf/io/parquet.hpp" namespace "cudf::io" nogil:
    cdef cppclass parquet_reader_options:
        ctypedef enum boolean_param_id \
                """cudf::io::parquet_reader_options::"
                boolean_param_id""":
            STRINGS_TO_CATEGORICAL \
                """cudf::io::parquet_reader_options::
                boolean_param_id::STRINGS_TO_CATEGORICAL"""
            USE_PANDAS_METADATA \
                """cudf::io::parquet_reader_options::
                boolean_param_id::USE_PANDAS_METADATA"""

        ctypedef enum size_type_param_id \
                """cudf::io::parquet_reader_options::
                size_type_param_id""":
            SKIP_ROWS \
                """cudf::io::parquet_reader_options::
                size_type_param_id::SKIP_ROWS"""
            NUM_ROWS \
                """cudf::io::parquet_reader_options::
                    size_type_param_id::NUM_ROWS"""

        parquet_reader_options() except +
        cudf_io_types.source_info source_info() except +
        vector[string] columns() except +
        vector[vector[size_type]] row_groups() except +
        data_type timestamp_type() except +
        bool get(boolean_param_id id) except +
        size_type get(size_type_param_id id) except +

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
        parquet_reader_options_builder& set(
            parquet_reader_options.boolean_param_id param_id,
            bool val
        ) except +
        parquet_reader_options_builder& set(
            parquet_reader_options.size_type_param_id  param_id,
            size_type val
        ) except +
        parquet_reader_options_builder& timestamp_type(
            data_type type
        ) except +
        parquet_reader_options build() except +

    cdef cudf_io_types.table_with_metadata read_parquet(
        parquet_reader_options args) except +

    cdef cppclass parquet_writer_options:
        parquet_writer_options() except +
        cudf_io_types.sink_info sink_info() except +
        cudf_io_types.compression_type compression() except +
        cudf_io_types.statistics_freq stats_level() except +
        cudf_table_view.table_view table() except +
        const cudf_io_types.table_metadata metadata() except +
        bool is_filemetadata_required() except +
        string metadata_out_file_path() except+

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
        parquet_writer_options_builder& metadata(
            cudf_io_types.table_metadata *m
        ) except +
        parquet_writer_options_builder& stats_level(
            cudf_io_types.statistics_freq sf
        ) except +
        parquet_writer_options_builder& compression(
            cudf_io_types.compression_type compression
        ) except +
        parquet_writer_options_builder& filemetadata_required(
            bool req
        ) except +
        parquet_writer_options_builder& metadata_out_file_path(
            string metadata_out_file_path
        ) except +

        parquet_writer_options build() except +

    cdef unique_ptr[vector[uint8_t]] \
        write_parquet(parquet_writer_options args) except +

    cdef cppclass chunked_parquet_writer_options:
        chunked_parquet_writer_options() except +
        cudf_io_types.sink_info sink_info() except +
        cudf_io_types.compression_type compression() except +
        cudf_io_types.statistics_freq stats_level() except +
        cudf_io_types.table_metadata_with_nullability* nullable_metadata(
        ) except+

        @staticmethod
        chunked_parquet_writer_options_builder builder(
            cudf_io_types.sink_info sink_,
        ) except +

    cdef cppclass chunked_parquet_writer_options_builder:
        chunked_parquet_writer_options_builder() except +
        chunked_parquet_writer_options_builder(
            cudf_io_types.sink_info sink_,
        ) except +
        chunked_parquet_writer_options_builder& nullable_metadata(
            cudf_io_types.table_metadata_with_nullability *m
        ) except +
        chunked_parquet_writer_options_builder& stats_level(
            cudf_io_types.statistics_freq sf
        ) except +
        chunked_parquet_writer_options_builder& compression(
            cudf_io_types.compression_type compression
        ) except +

        chunked_parquet_writer_options build() except +

    cdef shared_ptr[pq_chunked_state] write_parquet_chunked_begin(
        chunked_parquet_writer_options args
    ) except +

    cdef void write_parquet_chunked(cudf_table_view.table_view table_,
                                    shared_ptr[pq_chunked_state]) except +

    cdef unique_ptr[vector[uint8_t]] write_parquet_chunked_end(
        shared_ptr[pq_chunked_state],
        bool return_meta,
        string metadata_out_file_path,
    ) except +

cdef extern from "cudf/io/parquet.hpp" \
        namespace "cudf::io::detail::parquet" nogil:

    cdef cppclass pq_chunked_state:
        pass
