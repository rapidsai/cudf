# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *
from cudf._libxx.lib cimport *

from libcpp.string cimport string
from libcpp.vector cimport vector

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

    ctypedef enum compression_type:
        none "cudf::io::parquet::compression_type::none"
        snappy "cudf::io::parquet::compression_type::snappy"

    ctypedef enum statistics_freq:
        STATISTICS_NONE = 0
        STATISTICS_ROWGROUP = 1
        STATISTICS_PAGE = 2

    cdef cppclass writer_options:
        compression_type compression
        statistics_freq stats_granularity

        writer_options() except +

        writer_options(compression_type comp,
                       statistics_freq stats_lvl) except +

    cdef cppclass writer:
        writer(
            string filepath,
            const writer_options &args
        ) except +

        void write_all(const table_view &table_view) except +
