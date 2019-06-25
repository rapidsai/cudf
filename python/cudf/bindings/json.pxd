# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.io cimport *

from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "cudf.h" namespace "cudf::io::json" nogil:

    cdef struct reader_options:
        gdf_input_type  source_type;
        string          source;
        vector[string]  dtype;
        string          compression;
        bool            lines;

        reader_options() except +

    cdef cppclass reader:
        reader(const reader_options &args) except +

        cudf_table read() except +

        cudf_table read_byte_range(size_t offset, size_t size) except +
