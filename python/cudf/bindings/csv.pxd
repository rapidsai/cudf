# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.io cimport *
from cudf.bindings.types cimport table as cudf_table

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "cudf.h" namespace "cudf" nogil:

    # See cpp/include/cudf/io_types.h:33
    ctypedef gdf_input_type gdf_csv_input_form

    # See cpp/include/cudf/io_types.h:38
    ctypedef enum gdf_csv_quote_style:
        QUOTE_MINIMAL = 0,
        QUOTE_ALL,
        QUOTE_NONNUMERIC,
        QUOTE_NONE,

    # See cpp/include/cudf/io_types.h:62
    cdef struct csv_read_arg:
        gdf_csv_input_form  input_data_form
        string              filepath_or_buffer

        char                lineterminator
        char                delimiter
        bool                delim_whitespace
        bool                skipinitialspace

        gdf_size_type       nrows
        gdf_size_type       header

        vector[string]      names
        vector[string]      dtype

        int                 *index_col
        vector[int]         use_cols_indexes
        vector[string]      use_cols_names

        gdf_size_type       skiprows
        gdf_size_type       skipfooter

        bool                skip_blank_lines

        vector[string]      true_values
        vector[string]      false_values

        vector[string]      na_values
        bool                keep_default_na
        bool                na_filter

        string              prefix
        bool                mangle_dupe_cols

        bool                dayfirst

        string              compression
        char                thousands

        char                decimal

        char                quotechar
        gdf_csv_quote_style quoting
        bool                doublequote

        char                comment

        size_t              byte_range_offset
        size_t              byte_range_size

    cdef cudf_table read_csv(const csv_read_arg &args) except +
