# Copyright (c) 2024, NVIDIA CORPORATION.
from libc.stdint cimport int64_t
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.libcudf.io.types cimport source_info


cdef extern from "cudf/io/parquet_metadata.hpp" namespace "cudf::io" nogil:
    cdef cppclass parquet_column_schema:
        parquet_column_schema() except +libcudf_exception_handler
        string name() except +libcudf_exception_handler
        size_type num_children() except +libcudf_exception_handler
        parquet_column_schema child(int idx) except +libcudf_exception_handler
        vector[parquet_column_schema] children() except +libcudf_exception_handler

    cdef cppclass parquet_schema:
        parquet_schema() except +libcudf_exception_handler
        parquet_column_schema root() except +libcudf_exception_handler

    cdef cppclass parquet_metadata:
        parquet_metadata() except +libcudf_exception_handler
        parquet_schema schema() except +libcudf_exception_handler
        int64_t num_rows() except +libcudf_exception_handler
        size_type num_rowgroups() except +libcudf_exception_handler
        unordered_map[string, string] metadata() except +libcudf_exception_handler
        vector[unordered_map[string, int64_t]] rowgroup_metadata()\
            except +libcudf_exception_handler

    cdef parquet_metadata read_parquet_metadata(
        source_info src_info
    ) except +libcudf_exception_handler
