# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf._lib.cudf cimport *

from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "cudf/cudf.h" namespace "cudf::io::avro" nogil:

    cdef cppclass reader_options:
        vector[string] columns

        reader_options() except +

        reader_options(
            vector[string] columns
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

        cudf_table read_all() except +

        cudf_table read_rows(size_t skip_rows, size_t num_rows) except +
