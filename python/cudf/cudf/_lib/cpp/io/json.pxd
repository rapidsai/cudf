# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, unique_ptr
from libc.stdint cimport uint8_t

from cudf._lib.cpp.types cimport data_type, size_type
cimport cudf._lib.cpp.io.types as cudf_io_types
cimport cudf._lib.cpp.table.table_view as cudf_table_view

cdef extern from "cudf/io/json.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass json_reader_options:
        ctypedef enum boolean_param_id \
                """cudf::io::json_reader_options::"
                boolean_param_id""":
            LINES \
                """cudf::io::json_reader_options::
                boolean_param_id::LINES"""
            DAYFIRST \
                """cudf::io::json_reader_options::
                boolean_param_id::DAYFIRST"""

        ctypedef enum size_type_param_id \
                """cudf::io::json_reader_options::
                size_type_param_id""":
            BYTE_RANGE_OFFSET \
                """cudf::io::json_reader_options::
                size_type_param_id::BYTE_RANGE_OFFSET"""
            BYTE_RANGE_SIZE \
                """cudf::io::json_reader_options::
                    size_type_param_id::BYTE_RANGE_SIZE"""

        json_reader_options() except +
        cudf_io_types.source_info source_info() except +
        vector[string] dtypes() except +
        cudf_io_types.compression_type compression() except +
        bool get(boolean_param_id id) except +
        size_type get(size_type_param_id id) except +

        @staticmethod
        json_reader_options_builder builder(
            cudf_io_types.source_info src
        ) except +

    cdef cppclass json_reader_options_builder:
        json_reader_options_builder() except +
        json_reader_options_builder(
            cudf_io_types.source_info src
        ) except +
        json_reader_options_builder& dtypes(
            vector[string] types
        ) except +
        json_reader_options_builder& compression(
            cudf_io_types.compression_type compression
        ) except +
        json_reader_options_builder& set(
            json_reader_options.boolean_param_id param_id,
            bool val
        ) except +
        json_reader_options_builder& set(
            json_reader_options.size_type_param_id  param_id,
            size_type val
        ) except +

        json_reader_options build() except +

    cdef cudf_io_types.table_with_metadata read_json(
        json_reader_options &args) except +
