# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *
from cudf._lib.includes.io cimport *

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "cudf/cudf.h" namespace "cudf::io::csv" nogil:

    ctypedef enum quote_style:
        QUOTE_MINIMAL = 0,
        QUOTE_ALL,
        QUOTE_NONNUMERIC,
        QUOTE_NONE,

    cdef struct reader_options:
        string              compression
        char                lineterminator
        char                delimiter
        char                decimal
        char                thousands
        char                comment
        bool                dayfirst
        bool                delim_whitespace
        bool                skipinitialspace
        bool                skip_blank_lines
        gdf_size_type       header
        vector[string]      names
        vector[string]      dtype
        vector[int]         use_cols_indexes
        vector[string]      use_cols_names
        vector[int]         infer_date_indexes
        vector[string]      infer_date_names
        vector[string]      true_values
        vector[string]      false_values
        vector[string]      na_values
        bool                keep_default_na
        bool                na_filter
        string              prefix
        bool                mangle_dupe_cols
        char                quotechar
        quote_style         quoting
        bool                doublequote

        gdf_time_unit       out_time_unit

    cdef cppclass reader:
        reader(
            string filepath,
            const reader_options &options
        ) except +

        reader(
            const char *buffer,
            size_t length,
            const reader_options &options
        ) except +

        cudf_table read() except +

        cudf_table read_byte_range(size_t offset, size_t size) except +

        cudf_table read_rows(
            gdf_size_type num_skip_header,
            gdf_size_type num_skip_footer,
            gdf_size_type num_rows
        ) except +

cdef extern from "cudf/cudf.h" nogil:
    # See cpp/include/cudf/io_types.h:146
    ctypedef struct csv_write_arg:
        # Arguments to csv writer function
        gdf_column** columns
        int num_cols

        const char* filepath
        const char* line_terminator
        char delimiter

        const char* true_value
        const char* false_value
        const char* na_rep
        bool include_header

        int rows_per_chunk

    cdef gdf_error write_csv(csv_write_arg* args) except +
