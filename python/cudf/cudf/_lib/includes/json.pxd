# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *

from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "cudf/cudf.h" namespace "cudf::io::json" nogil:

    cdef cppclass reader_options:
        bool lines
        string compression
        vector[string] dtype

        reader_options() except +

        reader_options(
            bool lines,
            string compression,
            vector[string] dtype
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

        cudf_table read() except +

        cudf_table read_byte_range(size_t offset, size_t size) except +
