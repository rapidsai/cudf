# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from cudf._libxx.cpp.types cimport data_type, size_type
cimport cudf._libxx.cpp.io.types as cudf_io_types
cimport cudf._libxx.cpp.table.table_view as cudf_table_view


cdef extern from "cudf/io/functions.hpp" \
        namespace "cudf::experimental::io" nogil:

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

        # Reader settings
        cudf_io_types.compression_type compression
        size_t byte_range_offset
        size_t byte_range_size
        vector[string] names
        string prefix
        bool mangle_dupe_cols

        # Filter settings
        vector[string] use_cols_names
        vector[int] use_col_indexes
        size_t nrows
        size_t skiprows
        size_t skipfooter
        size_t header

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

    cdef cppclass read_orc_args:
        cudf_io_types.source_info source
        vector[string] columns
        size_type stripe
        size_type stripe_count
        size_type skip_rows
        size_type num_rows
        bool use_index
        bool use_np_dtypes
        data_type timestamp_type
        bool decimals_as_float
        int forced_decimals_scale

        read_orc_args() except +
        read_orc_args(cudf_io_types.source_info &src) except +

    cdef cudf_io_types.table_with_metadata read_orc(
        read_orc_args &args
    ) except +

    cdef cppclass read_parquet_args:
        cudf_io_types.source_info source
        vector[string] columns
        size_t row_group
        size_t row_group_count
        size_t skip_rows
        size_t num_rows
        bool strings_to_categorical
        bool use_pandas_metadata
        data_type timestamp_type

        read_parquet_args() except +
        read_parquet_args(cudf_io_types.source_info src) except +

    cdef cudf_io_types.table_with_metadata read_parquet(
        read_parquet_args args) except +

    cdef cppclass write_orc_args:
        cudf_io_types.sink_info sink
        cudf_io_types.compression_type compression
        bool enable_statistics
        cudf_table_view.table_view table
        const cudf_io_types.table_metadata *metadata

        write_orc_args() except +
        write_orc_args(cudf_io_types.sink_info sink_,
                       cudf_table_view.table_view table_,
                       cudf_io_types.table_metadata *metadata_,
                       cudf_io_types.compression_type compression_,
                       bool enable_statistics_) except +

    cdef void write_orc(write_orc_args args) except +

    cdef void write_parquet(write_parquet_args args) except +

    cdef cppclass write_parquet_args:
        cudf_io_types.sink_info sink
        cudf_io_types.compression_type compression
        cudf_io_types.statistics_freq stats_level
        cudf_table_view.table_view table
        const cudf_io_types.table_metadata *metadata

        write_parquet_args() except +
        write_parquet_args(cudf_io_types.sink_info sink_,
                           cudf_table_view.table_view table_,
                           cudf_io_types.table_metadata *table_metadata_,
                           cudf_io_types.compression_type compression_,
                           cudf_io_types.statistics_freq stats_lvl_) except +

    cdef void write_parquet(write_parquet_args args) except +
