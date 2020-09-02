# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, unique_ptr
from libc.stdint cimport uint8_t

from cudf._lib.cpp.types cimport data_type, size_type
cimport cudf._lib.cpp.io.types as cudf_io_types
cimport cudf._lib.cpp.table.table_view as cudf_table_view


cdef extern from "cudf/io/functions.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass read_avro_args:
        cudf_io_types.source_info source
        vector[string] columns
        size_type skip_rows
        size_type num_rows
        read_avro_args() except +
        read_avro_args(cudf_io_types.source_info &info) except +

    cdef cudf_io_types.table_with_metadata read_avro(
        read_avro_args &args) except +

    cdef cppclass read_json_args:
        cudf_io_types.source_info source
        bool lines
        cudf_io_types.compression_type compression
        vector[string] dtype
        bool dayfirst
        size_t byte_range_offset
        size_t byte_range_size

        read_json_args() except +

        read_json_args(cudf_io_types.source_info src) except +

    cdef cudf_io_types.table_with_metadata read_json(
        read_json_args &args) except +

    cdef cppclass read_csv_args:
        cudf_io_types.source_info source

        read_csv_args() except +
        read_csv_args(cudf_io_types.source_info src) except +

        # Reader settings
        cudf_io_types.compression_type compression
        size_t byte_range_offset
        size_t byte_range_size
        vector[string] names
        string prefix
        bool mangle_dupe_cols

        # Filter settings
        vector[string] use_cols_names
        vector[int] use_cols_indexes
        size_type nrows
        size_type skiprows
        size_type skipfooter
        size_type header

        # Parsing settings
        char lineterminator
        char delimiter
        char thousands
        char decimal
        char comment
        bool windowslinetermination
        bool delim_whitespace
        bool skipinitialspace
        bool skip_blank_lines
        cudf_io_types.quote_style quoting
        char quotechar
        bool doublequote
        vector[string] infer_date_names
        vector[int] infer_date_indexes

        # Conversion settings
        vector[string] dtype
        vector[string] true_values
        vector[string] false_values
        vector[string] na_values
        bool keep_default_na
        bool na_filter
        bool dayfirst

    cdef cudf_io_types.table_with_metadata read_csv(
        read_csv_args &args
    ) except +

    cdef cppclass read_parquet_args:
        cudf_io_types.source_info source
        vector[string] columns
        vector[vector[size_type]] row_groups
        size_t skip_rows
        size_t num_rows
        bool strings_to_categorical
        bool use_pandas_metadata
        data_type timestamp_type

        read_parquet_args() except +
        read_parquet_args(cudf_io_types.source_info src) except +

    cdef cudf_io_types.table_with_metadata read_parquet(
        read_parquet_args args) except +

    cdef cppclass write_csv_args:
        cudf_io_types.sink_info snk
        cudf_table_view.table_view table
        const cudf_io_types.table_metadata *metadata

        write_csv_args() except +
        write_csv_args(cudf_io_types.sink_info snk_,
                       cudf_table_view.table_view table_,
                       string na_,
                       bool include_header_,
                       int rows_per_chunk_,
                       string line_term_,
                       char delim_,
                       string true_v_,
                       string false_v_,
                       cudf_io_types.table_metadata *metadata_) except +

    cdef void write_csv(write_csv_args args) except +

    cdef cppclass write_parquet_args:
        cudf_io_types.sink_info sink
        cudf_io_types.compression_type compression
        cudf_io_types.statistics_freq stats_level
        cudf_table_view.table_view table
        const cudf_io_types.table_metadata *metadata
        bool return_filemetadata
        string metadata_out_file_path

        write_parquet_args() except +
        write_parquet_args(cudf_io_types.sink_info sink_,
                           cudf_table_view.table_view table_,
                           cudf_io_types.table_metadata *table_metadata_,
                           cudf_io_types.compression_type compression_,
                           cudf_io_types.statistics_freq stats_lvl_) except +

    cdef unique_ptr[vector[uint8_t]] \
        write_parquet(write_parquet_args args) except +

    cdef cppclass write_parquet_chunked_args:
        cudf_io_types.sink_info sink
        cudf_io_types.compression_type compression
        cudf_io_types.statistics_freq stats_level
        const cudf_io_types.table_metadata *metadata

        write_parquet_chunked_args() except +
        write_parquet_chunked_args(
            cudf_io_types.sink_info sink_,
            cudf_io_types.table_metadata *table_metadata_,
            cudf_io_types.compression_type compression_,
            cudf_io_types.statistics_freq stats_lvl_
        ) except +

    cdef shared_ptr[pq_chunked_state] \
        write_parquet_chunked_begin(write_parquet_chunked_args args) except +

    cdef void write_parquet_chunked(cudf_table_view.table_view table_,
                                    shared_ptr[pq_chunked_state]) except +

    cdef unique_ptr[vector[uint8_t]] write_parquet_chunked_end(
        shared_ptr[pq_chunked_state],
        bool return_meta,
        string metadata_out_file_path,
    ) except +

    cdef unique_ptr[vector[uint8_t]] merge_rowgroup_metadata(
        const vector[unique_ptr[vector[uint8_t]]]& metadata_list
    ) except +


cdef extern from "cudf/io/functions.hpp" \
        namespace "cudf::io::detail::parquet" nogil:

    cdef cppclass pq_chunked_state:
        pass
