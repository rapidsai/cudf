# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.io cimport *


cdef extern from "cudf.h" nogil:

    # See cpp/include/cudf/io_types.h:222
    ctypedef struct json_read_arg:
        # Output Arguments - Allocated in reader
        int             num_cols_out
        int             num_rows_out
        gdf_column      **data
        int             *index_col

        # Input arguments
        gdf_input_type  source_type;
        const char      *source;
        size_t          buffer_size;
        int             num_cols;
        const char      **dtype;
        char            *compression;
        bool            lines;
        size_t          byte_range_offset;
        size_t          byte_range_size;

    cdef gdf_error read_json(json_read_arg *args) except +