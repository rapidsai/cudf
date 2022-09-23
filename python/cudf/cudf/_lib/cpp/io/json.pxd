# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

cimport cudf._lib.cpp.io.types as cudf_io_types
cimport cudf._lib.cpp.table.table_view as cudf_table_view
from cudf._lib.cpp.types cimport data_type, size_type


cdef extern from "cudf/io/json.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass json_reader_options:
        json_reader_options() except+
        cudf_io_types.source_info get_source() except+
        vector[string] get_dtypes() except+
        cudf_io_types.compression_type get_compression() except +
        size_type get_byte_range_offset() except+
        size_type get_byte_range_size() except+
        bool is_enabled_lines() except+
        bool is_enabled_dayfirst() except+
        bool is_enabled_experimental() except+

        # setter
        void set_dtypes(vector[data_type] types) except+
        void set_dtypes(map[string, data_type] types) except+
        void set_compression(
            cudf_io_types.compression_type compression
        ) except+
        void set_byte_range_offset(size_type offset) except+
        void set_byte_range_size(size_type size) except+
        void enable_lines(bool val) except+
        void enable_dayfirst(bool val) except+
        void enable_experimental(bool val) except+

        @staticmethod
        json_reader_options_builder builder(
            cudf_io_types.source_info src
        ) except+

    cdef cppclass json_reader_options_builder:
        json_reader_options_builder() except+
        json_reader_options_builder(
            cudf_io_types.source_info src
        ) except+
        json_reader_options_builder& dtypes(
            vector[string] types
        ) except+
        json_reader_options_builder& dtypes(
            vector[data_type] types
        ) except+
        json_reader_options_builder& dtypes(
            map[string, data_type] types
        ) except+
        json_reader_options_builder& compression(
            cudf_io_types.compression_type compression
        ) except+
        json_reader_options_builder& byte_range_offset(
            size_type offset
        ) except+
        json_reader_options_builder& byte_range_size(
            size_type size
        ) except+
        json_reader_options_builder& lines(
            bool val
        ) except+
        json_reader_options_builder& dayfirst(
            bool val
        ) except+
        json_reader_options_builder& experimental(
            bool val
        ) except+

        json_reader_options build() except+

    cdef cudf_io_types.table_with_metadata read_json(
        json_reader_options &options) except+
