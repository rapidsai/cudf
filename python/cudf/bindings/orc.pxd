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

    cdef cppclass OrcReaderOptions:
        vector[string] columns
        bool use_index

        OrcReaderOptions() except +

        OrcReaderOptions(
            vector[string] columns,
            bool use_index
        ) except +

    cdef cppclass OrcReader:
        OrcReader(
            string filepath,
            const OrcReaderOptions &args
        ) except +

        OrcReader(
            const char *buffer,
            size_t length,
            const OrcReaderOptions &args
        ) except +

        cudf_table read_all() except +

        cudf_table read_rows(size_t skip_rows, size_t num_rows) except +

        cudf_table read_stripe(size_t stripe) except +
