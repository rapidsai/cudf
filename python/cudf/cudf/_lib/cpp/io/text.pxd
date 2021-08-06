# Copyright (c) 2020-2021, NVIDIA CORPORATION.

from libcpp.string cimport string

cimport cudf._lib.cpp.io.types as cudf_io_types
cimport cudf._lib.cpp.table.table_view as cudf_table_view
from cudf._lib.cpp.types cimport data_type, size_type


cdef extern from "cudf/io/text.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass text_reader_options:
        text_reader_options() except+

        cudf_io_types.source_info get_source() except+
        string get_delimiter() except+

        void set_delimiter(string delimiter) except+

        @staticmethod
        text_reader_options_builder builder(
            cudf_io_types.source_info src
        ) except+

    cdef cppclass text_reader_options_builder:
        text_reader_options_builder() except+
        text_reader_options_builder(cudf_io_types.source_info &src) except+

        text_reader_options_builder& delimiter(string delimiter) except+

        text_reader_options build() except+

    cdef cudf_io_types.table_with_metadata read_text(
        text_reader_options opts
    ) except +
