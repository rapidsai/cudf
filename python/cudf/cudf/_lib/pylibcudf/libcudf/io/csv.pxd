# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

cimport cudf._lib.pylibcudf.libcudf.io.types as cudf_io_types
cimport cudf._lib.pylibcudf.libcudf.table.table_view as cudf_table_view
from cudf._lib.pylibcudf.libcudf.types cimport data_type, size_type


cdef extern from "cudf/io/csv.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass csv_reader_options:
        csv_reader_options() except +

        # Getter

        cudf_io_types.source_info get_source() except +
        # Reader settings
        cudf_io_types.compression_type get_compression() except +
        size_t get_byte_range_offset() except +
        size_t get_byte_range_size() except +
        vector[string] get_names() except +
        string get_prefix() except +
        bool is_enabled_mangle_dupe_cols() except +

        # Filter settings
        vector[string] get_use_cols_names() except +
        vector[int] get_use_cols_indexes() except +
        size_type get_nrows() except +
        size_type get_skiprows() except +
        size_type get_skipfooter() except +
        size_type get_header() except +

        # Parsing settings
        char get_lineterminator() except +
        char get_delimiter() except +
        char get_thousands() except +
        char get_decimal() except +
        char get_comment() except +
        bool is_enabled_windowslinetermination() except +
        bool is_enabled_delim_whitespace() except +
        bool is_enabled_skipinitialspace() except +
        bool is_enabled_skip_blank_lines() except +
        cudf_io_types.quote_style get_quoting() except +
        char get_quotechar() except +
        bool is_enabled_doublequote() except +
        vector[string] get_parse_dates_names() except +
        vector[int] get_parse_dates_indexes() except +
        vector[string] get_parse_hex_names() except +
        vector[int] get_parse_hex_indexes() except +

        # Conversion settings
        vector[string] get_dtype() except +
        vector[string] get_true_values() except +
        vector[string] get_false_values() except +
        vector[string] get_na_values() except +
        bool is_enabled_keep_default_na() except +
        bool is_enabled_na_filter() except +
        bool is_enabled_dayfirst() except +

        # setter

        # Reader settings
        void set_compression(cudf_io_types.compression_type comp) except +
        void set_byte_range_offset(size_t val) except +
        void set_byte_range_size(size_t val) except +
        void set_names(vector[string] val) except +
        void set_prefix(string pfx) except +
        void set_mangle_dupe_cols(bool val) except +

        # Filter settings
        void set_use_cols_names(vector[string] col_names) except +
        void set_use_cols_indexes(vector[int] col_ind) except +
        void set_nrows(size_type n_rows) except +
        void set_skiprows(size_type val) except +
        void set_skipfooter(size_type val) except +
        void set_header(size_type hdr) except +

        # Parsing settings
        void set_lineterminator(char val) except +
        void set_delimiter(char val) except +
        void set_thousands(char val) except +
        void set_decimal(char val) except +
        void set_comment(char val) except +
        void enable_windowslinetermination(bool val) except +
        void enable_delim_whitespace(bool val) except +
        void enable_skipinitialspace(bool val) except +
        void enable_skip_blank_lines(bool val) except +
        void set_quoting(cudf_io_types.quote_style style) except +
        void set_quotechar(char val) except +
        void set_doublequote(bool val) except +
        void set_parse_dates(vector[string]) except +
        void set_parse_dates(vector[int]) except +
        void set_parse_hex(vector[string]) except +
        void set_parse_hex(vector[int]) except +

        # Conversion settings
        void set_dtypes(vector[data_type] types) except +
        void set_dtypes(map[string, data_type] types) except +
        void set_true_values(vector[string] vals) except +
        void set_false_values(vector[string] vals) except +
        void set_na_values(vector[string] vals) except +
        void enable_keep_default_na(bool val) except +
        void enable_na_filter(bool val) except +
        void enable_dayfirst(bool val) except +
        void set_timestamp_type(data_type type) except +

        @staticmethod
        csv_reader_options_builder builder(
            cudf_io_types.source_info src
        ) except +

    cdef cppclass csv_reader_options_builder:

        csv_reader_options_builder() except +
        csv_reader_options_builder(
            cudf_io_types.source_info src
        ) except +

        csv_reader_options_builder& source(
            cudf_io_types.source_info info
        ) except +
        # Reader settings
        csv_reader_options_builder& compression(
            cudf_io_types.compression_type comp
        ) except +
        csv_reader_options_builder& byte_range_offset(size_t val) except +
        csv_reader_options_builder& byte_range_size(size_t val) except +
        csv_reader_options_builder& names(vector[string] val) except +
        csv_reader_options_builder& prefix(string pfx) except +
        csv_reader_options_builder& mangle_dupe_cols(bool val) except +

        # Filter settings
        csv_reader_options_builder& use_cols_names(
            vector[string] col_names
        ) except +
        csv_reader_options_builder& use_cols_indexes(
            vector[int] col_ind
        ) except +
        csv_reader_options_builder& nrows(size_type n_rows) except +
        csv_reader_options_builder& skiprows(size_type val) except +
        csv_reader_options_builder& skipfooter(size_type val) except +
        csv_reader_options_builder& header(size_type hdr) except +

        # Parsing settings
        csv_reader_options_builder& lineterminator(char val) except +
        csv_reader_options_builder& delimiter(char val) except +
        csv_reader_options_builder& thousands(char val) except +
        csv_reader_options_builder& decimal(char val) except +
        csv_reader_options_builder& comment(char val) except +
        csv_reader_options_builder& windowslinetermination(bool val) except +
        csv_reader_options_builder& delim_whitespace(bool val) except +
        csv_reader_options_builder& skipinitialspace(bool val) except +
        csv_reader_options_builder& skip_blank_lines(bool val) except +
        csv_reader_options_builder& quoting(
            cudf_io_types.quote_style style
        ) except +
        csv_reader_options_builder& quotechar(char val) except +
        csv_reader_options_builder& doublequote(bool val) except +
        csv_reader_options_builder& parse_dates(vector[string]) except +
        csv_reader_options_builder& parse_dates(vector[int]) except +

        # Conversion settings
        csv_reader_options_builder& dtypes(vector[string] types) except +
        csv_reader_options_builder& dtypes(vector[data_type] types) except +
        csv_reader_options_builder& dtypes(
            map[string, data_type] types
        ) except +
        csv_reader_options_builder& true_values(vector[string] vals) except +
        csv_reader_options_builder& false_values(vector[string] vals) except +
        csv_reader_options_builder& na_values(vector[string] vals) except +
        csv_reader_options_builder& keep_default_na(bool val) except +
        csv_reader_options_builder& na_filter(bool val) except +
        csv_reader_options_builder& dayfirst(bool val) except +
        csv_reader_options_builder& timestamp_type(data_type type) except +

        csv_reader_options build() except +

    cdef cudf_io_types.table_with_metadata read_csv(
        csv_reader_options &options
    ) except +

    cdef cppclass csv_writer_options:
        csv_writer_options() except +

        cudf_io_types.sink_info get_sink() except +
        cudf_table_view.table_view get_table() except +
        cudf_io_types.table_metadata get_metadata() except +
        string get_na_rep() except +
        bool is_enabled_include_header() except +
        size_type get_rows_per_chunk() except +
        string get_line_terminator() except +
        char get_inter_column_delimiter() except +
        string get_true_value() except +
        string get_false_value() except +
        vector[string] get_names() except +

        # setter
        void set_metadata(cudf_io_types.table_metadata* val) except +
        void set_na_rep(string val) except +
        void enable_include_header(bool val) except +
        void set_rows_per_chunk(size_type val) except +
        void set_line_terminator(string term) except +
        void set_inter_column_delimiter(char delim) except +
        void set_true_value(string val) except +
        void set_false_value(string val) except +
        void set_names(vector[string] val) except +

        @staticmethod
        csv_writer_options_builder builder(
            cudf_io_types.sink_info sink,
            cudf_table_view.table_view table
        ) except +

    cdef cppclass csv_writer_options_builder:
        csv_writer_options_builder() except +
        csv_writer_options_builder(
            cudf_io_types.sink_info sink,
            cudf_table_view.table_view table
        ) except +

        csv_writer_options_builder& names(vector[string] val) except +
        csv_writer_options_builder& na_rep(string val) except +
        csv_writer_options_builder& include_header(bool val) except +
        csv_writer_options_builder& rows_per_chunk(size_type val) except +
        csv_writer_options_builder& line_terminator(string term) except +
        csv_writer_options_builder& inter_column_delimiter(char delim) except +
        csv_writer_options_builder& true_value(string val) except +
        csv_writer_options_builder& false_value(string val) except +

        csv_writer_options build() except +

    cdef void write_csv(csv_writer_options args) except +
