# Copyright (c) 2020-2024, NVIDIA CORPORATION.
cimport pylibcudf.libcudf.io.types as cudf_io_types
from libcpp.string cimport string
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/io/avro.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass avro_reader_options:
        avro_reader_options() except +libcudf_exception_handler
        cudf_io_types.source_info get_source() except +libcudf_exception_handler
        vector[string] get_columns() except +libcudf_exception_handler
        size_type get_skip_rows() except +libcudf_exception_handler
        size_type get_num_rows() except +libcudf_exception_handler

        # setters

        void set_columns(vector[string] col_names) except +libcudf_exception_handler
        void set_skip_rows(size_type val) except +libcudf_exception_handler
        void set_num_rows(size_type val) except +libcudf_exception_handler

        @staticmethod
        avro_reader_options_builder builder(
            cudf_io_types.source_info src
        ) except +libcudf_exception_handler

    cdef cppclass avro_reader_options_builder:
        avro_reader_options_builder() except +libcudf_exception_handler
        avro_reader_options_builder(
            cudf_io_types.source_info src
        ) except +libcudf_exception_handler
        avro_reader_options_builder& columns(
            vector[string] col_names
        ) except +libcudf_exception_handler
        avro_reader_options_builder& skip_rows(
            size_type val
        ) except +libcudf_exception_handler
        avro_reader_options_builder& num_rows(
            size_type val
        ) except +libcudf_exception_handler

        avro_reader_options build() except +libcudf_exception_handler

    cdef cudf_io_types.table_with_metadata read_avro(
        avro_reader_options &options
    ) except +libcudf_exception_handler
