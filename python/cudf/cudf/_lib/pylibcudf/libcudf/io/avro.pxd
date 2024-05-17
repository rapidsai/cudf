# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.string cimport string
from libcpp.vector cimport vector

cimport cudf._lib.pylibcudf.libcudf.io.types as cudf_io_types
from cudf._lib.pylibcudf.libcudf.types cimport size_type


cdef extern from "cudf/io/avro.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass avro_reader_options:
        avro_reader_options() except +
        cudf_io_types.source_info get_source() except +
        vector[string] get_columns() except +
        size_type get_skip_rows() except +
        size_type get_num_rows() except +

        # setters

        void set_columns(vector[string] col_names) except +
        void set_skip_rows(size_type val) except +
        void set_num_rows(size_type val) except +

        @staticmethod
        avro_reader_options_builder builder(
            cudf_io_types.source_info src
        ) except +

    cdef cppclass avro_reader_options_builder:
        avro_reader_options_builder() except +
        avro_reader_options_builder(
            cudf_io_types.source_info src
        ) except +
        avro_reader_options_builder& columns(vector[string] col_names) except +
        avro_reader_options_builder& skip_rows(size_type val) except +
        avro_reader_options_builder& num_rows(size_type val) except +

        avro_reader_options build() except +

    cdef cudf_io_types.table_with_metadata read_avro(
        avro_reader_options &options
    ) except +
