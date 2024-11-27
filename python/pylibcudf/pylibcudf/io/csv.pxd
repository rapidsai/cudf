# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from pylibcudf.libcudf.io.csv cimport (
    csv_writer_options,
    csv_writer_options_builder,
    csv_reader_options,
    csv_reader_options_builder,
)
from pylibcudf.io.types cimport SinkInfo, SourceInfo, TableWithMetadata
from pylibcudf.table cimport Table

from pylibcudf.libcudf.io.types cimport (
    compression_type,
    quote_style,
    table_with_metadata,
)
from pylibcudf.libcudf.types cimport size_type

cdef class CsvReaderOptions:
    cdef csv_reader_options c_obj
    cdef SourceInfo source
    cpdef void set_header(self, size_type header)
    cpdef void set_names(self, list col_names)
    cpdef void set_prefix(self, str prefix)
    cpdef void set_use_cols_indexes(self, list col_indices)
    cpdef void set_use_cols_names(self, list col_names)
    cpdef void set_delimiter(self, str delimiter)
    cpdef void set_thousands(self, str thousands)
    cpdef void set_comment(self, str comment)
    cpdef void set_parse_dates(self, list val)
    cpdef void set_parse_hex(self, list val)
    cpdef void set_dtypes(self, object types)
    cpdef void set_true_values(self, list true_values)
    cpdef void set_false_values(self, list false_values)
    cpdef void set_na_values(self, list na_values)


cdef class CsvReaderOptionsBuilder:
    cdef csv_reader_options_builder c_obj
    cdef SourceInfo source
    cpdef CsvReaderOptionsBuilder compression(self, compression_type compression)
    cpdef CsvReaderOptionsBuilder mangle_dupe_cols(self, bool mangle_dupe_cols)
    cpdef CsvReaderOptionsBuilder byte_range_offset(self, size_t byte_range_offset)
    cpdef CsvReaderOptionsBuilder byte_range_size(self, size_t byte_range_size)
    cpdef CsvReaderOptionsBuilder nrows(self, size_type nrows)
    cpdef CsvReaderOptionsBuilder skiprows(self, size_type skiprows)
    cpdef CsvReaderOptionsBuilder skipfooter(self, size_type skipfooter)
    cpdef CsvReaderOptionsBuilder quoting(self, quote_style quoting)
    cpdef CsvReaderOptionsBuilder lineterminator(self, str lineterminator)
    cpdef CsvReaderOptionsBuilder quotechar(self, str quotechar)
    cpdef CsvReaderOptionsBuilder decimal(self, str decimal)
    cpdef CsvReaderOptionsBuilder delim_whitespace(self, bool delim_whitespace)
    cpdef CsvReaderOptionsBuilder skipinitialspace(self, bool skipinitialspace)
    cpdef CsvReaderOptionsBuilder skip_blank_lines(self, bool skip_blank_lines)
    cpdef CsvReaderOptionsBuilder doublequote(self, bool doublequote)
    cpdef CsvReaderOptionsBuilder keep_default_na(self, bool keep_default_na)
    cpdef CsvReaderOptionsBuilder na_filter(self, bool na_filter)
    cpdef CsvReaderOptionsBuilder dayfirst(self, bool dayfirst)
    cpdef CsvReaderOptions build(self)

cpdef TableWithMetadata read_csv(CsvReaderOptions options)

cdef class CsvWriterOptions:
    cdef csv_writer_options c_obj
    cdef Table table
    cdef SinkInfo sink


cdef class CsvWriterOptionsBuilder:
    cdef csv_writer_options_builder c_obj
    cdef Table table
    cdef SinkInfo sink
    cpdef CsvWriterOptionsBuilder names(self, list names)
    cpdef CsvWriterOptionsBuilder na_rep(self, str val)
    cpdef CsvWriterOptionsBuilder include_header(self, bool val)
    cpdef CsvWriterOptionsBuilder rows_per_chunk(self, int val)
    cpdef CsvWriterOptionsBuilder line_terminator(self, str term)
    cpdef CsvWriterOptionsBuilder inter_column_delimiter(self, str delim)
    cpdef CsvWriterOptionsBuilder true_value(self, str val)
    cpdef CsvWriterOptionsBuilder false_value(self, str val)
    cpdef CsvWriterOptions build(self)


cpdef void write_csv(CsvWriterOptions options)
