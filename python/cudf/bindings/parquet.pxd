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

    # See cpp/include/cudf/io_types.h
    cdef cppclass ParquetReaderOptions:
        vector[string] columns
        bool strings_to_categorical

        ParquetReaderOptions() except +

        ParquetReaderOptions(
            vector[string] columns,
            bool strings_to_categorical
        ) except +

    cdef cppclass ParquetReader:
        ParquetReader(
            string filepath,
            const ParquetReaderOptions &args
        ) except +

        ParquetReader(
            const char *buffer,
            size_t length,
            const ParquetReaderOptions &args
        ) except +

        string get_index_column() except +

        cudf_table read_all() except +

        cudf_table read_rows(size_t skip_rows, size_t num_rows) except +

        cudf_table read_row_group(size_t row_group) except +
