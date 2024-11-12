# Copyright (c) 2024, NVIDIA CORPORATION.

from libc.stdint cimport int64_t
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.libcudf.io.types cimport source_info


cdef extern from "cudf/io/parquet_metadata.hpp" namespace "cudf::io" nogil:
    cdef cppclass parquet_column_schema:
        parquet_column_schema() except+
        string name() except+
        size_type num_children() except+
        parquet_column_schema child(int idx) except+
        vector[parquet_column_schema] children() except+

    cdef cppclass parquet_schema:
        parquet_schema() except+
        parquet_column_schema root() except+

    cdef cppclass parquet_metadata:
        parquet_metadata() except+
        parquet_schema schema() except+
        int64_t num_rows() except+
        size_type num_rowgroups() except+
        unordered_map[string, string] metadata() except+
        vector[unordered_map[string, int64_t]] rowgroup_metadata() except+

    cdef parquet_metadata read_parquet_metadata(source_info src_info) except+
