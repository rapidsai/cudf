# Copyright (c) 2025, NVIDIA CORPORATION.

from libc.stdint cimport int32_t, int64_t, uint8_t
from libcpp cimport bool
from libcpp.optional cimport optional
from libcpp.string cimport string
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler


cdef extern from "cudf/io/parquet_schema.hpp" namespace "cudf::io::parquet" nogil:
    cdef cppclass FileMetaData:
        FileMetaData() except +libcudf_exception_handler
        int32_t version
        int64_t num_rows
        string created_by
