# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.string cimport string
from libcpp.vector cimport vector

from cudf._lib.cpp.types cimport size_type
cimport cudf._lib.cpp.io.types as cudf_io_types


cdef extern from "cudf/io/avro.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass avro_reader_options:
        avro_reader_options() except +
        ctypedef enum size_type_param_id \
                """cudf::io::avro_reader_options::
                size_type_param_id""":
            SKIP_ROWS \
                """cudf::io::avro_reader_options::
                size_type_param_id::SKIP_ROWS"""
            NUM_ROWS \
                """cudf::io::avro_reader_options::
                    size_type_param_id::NUM_ROWS"""

        cudf_io_types.source_info source_info() except +
        vector[string] columns() except +
        size_type get(size_type_param_id id) except +

        @staticmethod
        avro_reader_options_builder builder(
            cudf_io_types.source_info src
        ) except +

    cdef cppclass avro_reader_options_builder:
        avro_reader_options_builder() except +
        avro_reader_options_builder(
            cudf_io_types.source_info src
        ) except +
        avro_reader_options_builder& columns(
            vector[string] col_names
        ) except +
        avro_reader_options_builder& set(
            avro_reader_options.size_type_param_id  param_id,
            size_type val
        ) except +

        avro_reader_options build() except +

    cdef cudf_io_types.table_with_metadata read_avro(
        avro_reader_options &args) except +
