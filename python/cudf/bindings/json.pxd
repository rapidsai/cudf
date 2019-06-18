# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.io cimport *

from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "cudf.h" namespace "cudf" nogil:

    # See TODO
    cdef struct json_reader_args:
        gdf_input_type  source_type;
        string          source;
        vector[string]  dtype;
        string          compression;
        bool            lines;

        json_reader_args() except +

    cdef cppclass JsonReader:

        JsonReader()

        JsonReader(const json_reader_args &args) except +

        cudf_table read() except +

        cudf_table read_byte_range(size_t offset, size_t size) except +
