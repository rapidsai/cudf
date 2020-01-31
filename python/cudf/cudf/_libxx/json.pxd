# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._libxx.lib cimport *
from cudf._libxx.io_types cimport *

from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "cudf/io/readers.hpp" \
        namespace "cudf::experimental::io::detail::json" nogil:

    cdef cppclass reader_options:
        bool lines
        compression_type compression
        vector[string] dtype
        bool dayfirst

        reader_options() except +

        reader_options(
            bool lines,
            compression_type compression,
            vector[string] dtype,
            bool dayfirst
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

        table_with_metadata read_all() except +

        table_with_metadata read_byte_range(
            size_t offset, size_t size) except +
