# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

cimport cudf._lib.cpp.io.types as cudf_io_types
cimport cudf._lib.cpp.table.table_view as cudf_table_view
from cudf._lib.cpp.libcpp.optional cimport optional
from cudf._lib.cpp.types cimport data_type, size_type


cdef extern from "cudf/io/orc.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass orc_reader_options:
        orc_reader_options() except +

        cudf_io_types.source_info get_source() except +
        vector[vector[size_type]] get_stripes() except +
        size_type get_skip_rows() except +
        size_type get_num_rows() except +
        bool is_enabled_use_index() except +
        bool is_enabled_use_np_dtypes() except +
        data_type get_timestamp_type() except +
        bool is_enabled_decimals_as_float64() except +
        int get_forced_decimals_scale() except +

        void set_columns(vector[string] col_names) except +
        void set_stripes(vector[vector[size_type]] strps) except +
        void set_skip_rows(size_type rows) except +
        void set_num_rows(size_type nrows) except +
        void enable_use_index(bool val) except +
        void enable_use_np_dtypes(bool val) except +
        void set_timestamp_type(data_type type) except +

        @staticmethod
        orc_reader_options_builder builder(
            cudf_io_types.source_info src
        ) except +

    cdef cppclass orc_reader_options_builder:
        orc_reader_options_builder() except +
        orc_reader_options_builder(cudf_io_types.source_info &src) except +

        orc_reader_options_builder& columns(vector[string] col_names) except +
        orc_reader_options_builder& \
            stripes(vector[vector[size_type]] strps) except +
        orc_reader_options_builder& skip_rows(size_type rows) except +
        orc_reader_options_builder& num_rows(size_type nrows) except +
        orc_reader_options_builder& use_index(bool val) except +
        orc_reader_options_builder& use_np_dtypes(bool val) except +
        orc_reader_options_builder& timestamp_type(data_type type) except +

        orc_reader_options build() except +

    cdef cudf_io_types.table_with_metadata read_orc(
        orc_reader_options opts
    ) except +

    cdef cppclass orc_writer_options:
        orc_writer_options()
        cudf_io_types.sink_info get_sink() except +
        cudf_io_types.compression_type get_compression() except +
        bool is_enabled_statistics() except +
        size_t get_stripe_size_bytes() except +
        size_type get_stripe_size_rows() except +
        size_type get_row_index_stride() except +
        cudf_table_view.table_view get_table() except +
        const optional[cudf_io_types.table_input_metadata]& get_metadata(
        ) except +

        # setter
        void set_compression(cudf_io_types.compression_type comp) except +
        void enable_statistics(bool val) except +
        void set_stripe_size_bytes(size_t val) except +
        void set_stripe_size_rows(size_type val) except +
        void set_row_index_stride(size_type val) except +
        void set_table(cudf_table_view.table_view tbl) except +
        void set_metadata(cudf_io_types.table_input_metadata meta) except +
        void set_key_value_metadata(map[string, string] kvm) except +

        @staticmethod
        orc_writer_options_builder builder(
            cudf_io_types.sink_info &sink,
            cudf_table_view.table_view &tbl
        ) except +

    cdef cppclass orc_writer_options_builder:
        # setter
        orc_writer_options_builder& compression(
            cudf_io_types.compression_type comp
        ) except +
        orc_writer_options_builder& enable_statistics(bool val) except +
        orc_writer_options_builder& stripe_size_bytes(size_t val) except +
        orc_writer_options_builder& stripe_size_rows(size_type val) except +
        orc_writer_options_builder& row_index_stride(size_type val) except +
        orc_writer_options_builder& table(
            cudf_table_view.table_view tbl
        ) except +
        orc_writer_options_builder& metadata(
            cudf_io_types.table_input_metadata meta
        ) except +
        orc_writer_options_builder& key_value_metadata(
            map[string, string] kvm
        ) except +

        orc_writer_options build() except +

    cdef void write_orc(orc_writer_options options) except +

    cdef cppclass chunked_orc_writer_options:
        chunked_orc_writer_options() except +
        cudf_io_types.sink_info get_sink() except +
        cudf_io_types.compression_type get_compression() except +
        bool enable_statistics() except +
        size_t stripe_size_bytes() except +
        size_type stripe_size_rows() except +
        size_type row_index_stride() except +
        cudf_table_view.table_view get_table() except +
        const optional[cudf_io_types.table_input_metadata]& get_metadata(
        ) except +

        # setter
        void set_compression(cudf_io_types.compression_type comp) except +
        void enable_statistics(bool val) except +
        void set_stripe_size_bytes(size_t val) except +
        void set_stripe_size_rows(size_type val) except +
        void set_row_index_stride(size_type val) except +
        void set_table(cudf_table_view.table_view tbl) except +
        void set_metadata(
            cudf_io_types.table_input_metadata meta
        ) except +
        void set_key_value_metadata(map[string, string] kvm) except +

        @staticmethod
        chunked_orc_writer_options_builder builder(
            cudf_io_types.sink_info &sink
        ) except +

    cdef cppclass chunked_orc_writer_options_builder:
        # setter
        chunked_orc_writer_options_builder& compression(
            cudf_io_types.compression_type comp
        ) except +
        chunked_orc_writer_options_builder& enable_statistics(
            bool val
        ) except +
        orc_writer_options_builder& stripe_size_bytes(size_t val) except +
        orc_writer_options_builder& stripe_size_rows(size_type val) except +
        orc_writer_options_builder& row_index_stride(size_type val) except +
        chunked_orc_writer_options_builder& table(
            cudf_table_view.table_view tbl
        ) except +
        chunked_orc_writer_options_builder& metadata(
            cudf_io_types.table_input_metadata meta
        ) except +
        chunked_orc_writer_options_builder& key_value_metadata(
            map[string, string] kvm
        ) except +

        chunked_orc_writer_options build() except +

    cdef cppclass orc_chunked_writer:
        orc_chunked_writer() except +
        orc_chunked_writer(chunked_orc_writer_options args) except +
        orc_chunked_writer& write(
            cudf_table_view.table_view table_,
        ) except +
        void close() except +
