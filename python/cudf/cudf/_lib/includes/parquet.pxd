# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map

from cudf._lib.cudf cimport *
from cudf._libxx.cpp.table.table_view cimport table_view


cdef extern from "cudf/cudf.h" namespace "cudf::io::parquet" nogil:

    cdef cppclass reader_options:
        vector[string] columns
        bool strings_to_categorical
        bool use_pandas_metadata
        gdf_time_unit timestamp_unit

        reader_options() except +

        reader_options(
            vector[string] columns,
            bool strings_as_category,
            bool use_pandas_metadata,
            gdf_time_unit timestamp_unit
        ) except +

    cdef cppclass reader:
        reader(
            string filepath,
            const reader_options &args
        ) except +

        reader(
            const char *buffer,
            size_t length,
            const reader_options &args
        ) except +

        string get_index_column() except +

        cudf_table read_all() except +

        cudf_table read_rows(size_t skip_rows, size_t num_rows) except +

        cudf_table read_row_group(size_t row_group) except +

cdef extern from "cudf/io/functions.hpp" \
        namespace "cudf::experimental::io" nogil:

    ctypedef enum compression_type:
        NONE "cudf::experimental::io::compression_type::NONE"
        SNAPPY "cudf::experimental::io::compression_type::SNAPPY"

    ctypedef enum statistics_freq:
        STATISTICS_NONE \
            "cudf::experimental::io::statistics_freq::STATISTICS_NONE"
        STATISTICS_ROWGROUP \
            "cudf::experimental::io::statistics_freq::STATISTICS_ROWGROUP"
        STATISTICS_PAGE \
            "cudf::experimental::io::statistics_freq::STATISTICS_PAGE"

    ctypedef enum io_type:
        FILEPATH \
            "cudf::experimental::io::io_types::FILEPATH"
        HOST_BUFFER \
            "cudf::experimental::io::io_types::HOST_BUFFER"
        ARROW_RANDOM_ACCESS_FILE \
            "cudf::experimental::io::io_types::ARROW_RANDOM_ACCESS_FILE"

    cdef cppclass sink_info:
        io_type type
        string filepath

        sink_info() except +
        sink_info(string file_path) except +

    cdef cppclass table_metadata:
        table_metadata() except +

        vector[string] column_names
        map[string, string] user_data

    cdef cppclass write_parquet_args:
        sink_info sink
        compression_type compression
        statistics_freq stats_level
        table_view table
        const table_metadata *metadata

        write_parquet_args() except +
        write_parquet_args(sink_info sink_,
                           table_view table_,
                           table_metadata *table_metadata_,
                           compression_type compression_,
                           statistics_freq stats_lvl_) except +

    cdef void write_parquet(write_parquet_args args)
