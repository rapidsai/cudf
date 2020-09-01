# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr, unique_ptr
from libc.stdint cimport uint8_t

from cudf._lib.cpp.types cimport data_type, size_type
cimport cudf._lib.cpp.io.types as cudf_io_types
cimport cudf._lib.cpp.table.table_view as cudf_table_view

cdef extern from "cudf/io/csv.hpp" \
        namespace "cudf::io" nogil:

    cdef cppclass csv_reader_options:
        csv_reader_options() except +

        # Getter

        cudf_io_types.source_info source() except+
        # Reader settings
        cudf_io_types.compression_type compression() except+
        size_t byte_range_offset() except+
        size_t byte_range_size() except+
        vector[string] names() except+
        string prefix() except+
        bool mangle_dupe_cols() except+

        # Filter settings
        vector[string] use_cols_names() except+
        vector[int] use_cols_indexes() except+
        size_type nrows() except+
        size_type skiprows() except+
        size_type skipfooter() except+
        size_type header() except+

        # Parsing settings
        char lineterminator() except+
        char delimiter() except+
        char thousands() except+
        char decimal() except+
        char comment() except+
        bool windowslinetermination() except+
        bool delim_whitespace() except+
        bool skipinitialspace() except+
        bool skip_blank_lines() except+
        cudf_io_types.quote_style quoting() except+
        char quotechar() except+
        bool doublequote() except+
        vector[string] infer_date_names() except+
        vector[int] infer_date_indexes() except+

        # Conversion settings
        vector[string] dtype() except+
        vector[string] true_values() except+
        vector[string] false_values() except+
        vector[string] na_values() except+
        bool keep_default_na() except+
        bool na_filter() except+
        bool dayfirst() except+

        cudf_io_types.source_info source() except+
        # Reader settings
        cudf_io_types.compression_type compression() except+
        size_t byte_range_offset() except+
        size_t byte_range_size() except+
        vector[string] names() except+
        string prefix() except+
        bool mangle_dupe_cols() except+

        # Filter settings
        vector[string] use_cols_names() except+
        vector[int] use_cols_indexes() except+
        size_type nrows() except+
        size_type skiprows() except+
        size_type skipfooter() except+
        size_type header() except+

        # Parsing settings
        char lineterminator() except+
        char delimiter() except+
        char thousands() except+
        char decimal() except+
        char comment() except+
        bool windowslinetermination() except+
        bool delim_whitespace() except+
        bool skipinitialspace() except+
        bool skip_blank_lines() except+
        cudf_io_types.quote_style quoting() except+
        char quotechar() except+
        bool doublequote() except+
        vector[string] infer_date_names() except+
        vector[int] infer_date_indexes() except+

        # Conversion settings
        vector[string] dtype() except+
        vector[string] true_values() except+
        vector[string] false_values() except+
        vector[string] na_values() except+
        bool keep_default_na() except+
        bool na_filter() except+
        bool dayfirst() except+
        data_type timestamp_type() except+

        # setter

        void source(cudf_io_types.source_info info) except+
        # Reader settings
        void compression(cudf_io_types.compression_type comp) except+
        void byte_range_offset(size_type val) except+
        void byte_range_size(size_type val) except+
        void names(vector[string] val) except+
        void prefix(string pfx) except+
        void mangle_dupe_cols(bool val) except+

        # Filter settings
        void use_cols_names(vector[string] col_names) except+
        void use_cols_indexes(vector[int] col_ind) except+
        void nrows(size_type n_rows) except+
        void skiprows(size_type val) except+
        void skipfooter(size_type val) except+
        void header(size_type hdr) except+

        # Parsing settings
        void lineterminator(char val) except+
        void delimiter(char val) except+
        void thousands(char val) except+
        void decimal(char val) except+
        void comment(char val) except+
        void windowslinetermination(bool val) except+
        void delim_whitespace(bool val) except+
        void skipinitialspace(bool val) except+
        void skip_blank_lines(bool val) except+
        void quoting(cudf_io_types.quote_style style) except+
        void quotechar(char val) except+
        void doublequote(bool val) except+
        void infer_date_names(vector[string]) except+
        void infer_date_indexes(vector[int]) except+

        # Conversion settings
        void dtypes(vector[string] types) except+
        void true_values(vector[string] vals) except+
        void false_values(vector[string] vals) except+
        void na_values(vector[string] vals) except+
        void keep_default_na(bool val) except+
        void na_filter(bool val) except+
        void dayfirst(bool val) except+
        void timestamp_type(data_type type) except+

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
        ) except+
        # Reader settings
        csv_reader_options_builder& compression(
            cudf_io_types.compression_type comp
        ) except+
        csv_reader_options_builder& byte_range_offset(size_type val) except+
        csv_reader_options_builder& byte_range_size(size_type val) except+
        csv_reader_options_builder& names(vector[string] val) except+
        csv_reader_options_builder& prefix(string pfx) except+
        csv_reader_options_builder& mangle_dupe_cols(bool val) except+

        # Filter settings
        csv_reader_options_builder& use_cols_names(
            vector[string] col_names
        ) except+
        csv_reader_options_builder& use_cols_indexes(
            vector[int] col_ind
        ) except+
        csv_reader_options_builder& nrows(size_type n_rows) except+
        csv_reader_options_builder& skiprows(size_type val) except+
        csv_reader_options_builder& skipfooter(size_type val) except+
        csv_reader_options_builder& header(size_type hdr) except+

        # Parsing settings
        csv_reader_options_builder& lineterminator(char val) except+
        csv_reader_options_builder& delimiter(char val) except+
        csv_reader_options_builder& thousands(char val) except+
        csv_reader_options_builder& decimal(char val) except+
        csv_reader_options_builder& comment(char val) except+
        csv_reader_options_builder& windowslinetermination(bool val) except+
        csv_reader_options_builder& delim_whitespace(bool val) except+
        csv_reader_options_builder& skipinitialspace(bool val) except+
        csv_reader_options_builder& skip_blank_lines(bool val) except+
        csv_reader_options_builder& quoting(
            cudf_io_types.quote_style style
        ) except+
        csv_reader_options_builder& quotechar(char val) except+
        csv_reader_options_builder& doublequote(bool val) except+
        csv_reader_options_builder& infer_date_names(vector[string]) except+
        csv_reader_options_builder& infer_date_indexes(vector[int]) except+

        # Conversion settings
        csv_reader_options_builder& dtypes(vector[string] types) except+
        csv_reader_options_builder& true_values(vector[string] vals) except+
        csv_reader_options_builder& false_values(vector[string] vals) except+
        csv_reader_options_builder& na_values(vector[string] vals) except+
        csv_reader_options_builder& keep_default_na(bool val) except+
        csv_reader_options_builder& na_filter(bool val) except+
        csv_reader_options_builder& dayfirst(bool val) except+
        csv_reader_options_builder& timestamp_type(data_type type) except+

        csv_reader_options build() except+

    cdef cudf_io_types.table_with_metadata read_csv(
        csv_reader_options &options
    ) except +

    cdef cppclass csv_writer_options:
        csv_writer_options() except+

        cudf_io_types.sink_info sink() except+
        cudf_table_view.table_view table() except+
        cudf_io_types.table_metadata metadata() except+
        string na_rep() except+
        bool include_header() except+
        size_type rows_per_chunk() except+
        string line_terminator() except+
        char inter_column_delimiter() except+
        string true_value() except+
        string false_value() except+

        # setter
        void metadata(cudf_io_types.table_metadata* val) except+
        void na_rep(string val) except+
        void include_header(bool val) except+
        void rows_per_chunk(size_type val) except+
        void line_terminator(string term) except+
        void inter_column_delimiter(char delim) except+
        void true_value(string val) except+
        void false_value(string val) except+

        @staticmethod
        csv_writer_options_builder builder(
            cudf_io_types.sink_info sink,
            cudf_table_view.table_view table
        ) except+

    cdef cppclass csv_writer_options_builder:
        csv_writer_options_builder() except+
        csv_writer_options_builder(
            cudf_io_types.sink_info sink,
            cudf_table_view.table_view table
        ) except+

        csv_writer_options_builder& metadata(
            cudf_io_types.table_metadata* val
        ) except+
        csv_writer_options_builder& na_rep(string val) except+
        csv_writer_options_builder& include_header(bool val) except+
        csv_writer_options_builder& rows_per_chunk(size_type val) except+
        csv_writer_options_builder& line_terminator(string term) except+
        csv_writer_options_builder& inter_column_delimiter(char delim) except+
        csv_writer_options_builder& true_value(string val) except+
        csv_writer_options_builder& false_value(string val) except+

        csv_writer_options build() except+

    cdef void write_csv(csv_writer_options args) except +
