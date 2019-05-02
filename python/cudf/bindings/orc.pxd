# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.io cimport *


cdef extern from "cudf.h" nogil:

    # See cpp/include/cudf/io_types.h
    ctypedef struct orc_read_arg:
        # Output Arguments - Allocated in reader
        int num_cols_out
        int num_rows_out
        gdf_column **data

        # Input arguments
        gdf_input_type source_type
        const char *source
        size_t buffer_size
        const char **use_cols
        int use_cols_len
        int skip_rows
        int num_rows

    cdef gdf_error read_orc(orc_read_arg *args) except +
