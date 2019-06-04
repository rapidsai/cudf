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

    # See TODO
    cdef struct json_read_arg:
        gdf_input_type  source_type;
        string          source;
        vector[string]  dtype;
        string          compression;
        bool            lines;
        size_t          byte_range_offset;
        size_t          byte_range_size;

        json_read_arg() except +

    cdef cudf_table read_json(json_read_arg &args) except +
