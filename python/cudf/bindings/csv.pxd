# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *

cdef char* _wrap_string(text)
cpdef is_file_like(obj)

cdef extern from "cudf.h" nogil:

    # See cpp/include/cudf/io_types.h:22
    ctypedef enum gdf_input_type:
        FILE_PATH = 0,
        HOST_BUFFER,

    # See cpp/include/cudf/io_types.h:33
    ctypedef gdf_input_type gdf_csv_input_form

    # See cpp/include/cudf/io_types.h:38
    ctypedef enum gdf_csv_quote_style:
        QUOTE_MINIMAL = 0,
        QUOTE_ALL,
        QUOTE_NONNUMERIC,
        QUOTE_NONE,

    # See cpp/include/cudf/io_types.h:62
    ctypedef struct csv_read_arg:
        # Output Arguments - Allocated in reader
        int                 num_cols_out
        int                 num_rows_out
        gdf_column          **data

        # Input Arguments - All data is in the host memory
        gdf_csv_input_form  input_data_form
        const char          *filepath_or_buffer
        size_t              buffer_size

        bool                windowslinetermination
        char                lineterminator
        char                delimiter
        bool                delim_whitespace
        bool                skipinitialspace

        gdf_size_type       nrows
        gdf_size_type       header

        int                 num_cols
        const char          **names
        const char          **dtype

        int                 *index_col
        int                 *use_cols_int
        int                 use_cols_int_len
        const char          **use_cols_char
        int                 use_cols_char_len

        gdf_size_type       skiprows
        gdf_size_type       skipfooter

        bool                skip_blank_lines

        const char          **true_values
        int                 num_true_values
        const char          **false_values
        int                 num_false_values

        const char          **na_values
        int                 num_na_values
        bool                keep_default_na
        bool                na_filter

        char                *prefix
        bool                mangle_dupe_cols

        bool                parse_dates
        bool                infer_datetime_format
        bool                dayfirst

        char                *compression
        char                thousands

        char                decimal

        char                quotechar
        gdf_csv_quote_style quoting
        bool                doublequote

        char                escapechar

        char                comment

        char                *encoding

        size_t              byte_range_offset
        size_t              byte_range_size

    cdef gdf_error read_csv(csv_read_arg *args) except +
