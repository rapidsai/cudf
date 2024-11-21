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
from pylibcudf.io.types cimport SinkInfo, SourceInfo
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

cdef class CsvReaderOptionsBuilder:
    cdef csv_reader_options_builder c_obj
    cdef SourceInfo source
    cdef CsvReaderOptionsBuilder compression(self, compression_type compression)
    cdef CsvReaderOptionsBuilder mangle_dupe_cols(self, bool mangle_dupe_cols)
    cdef CsvReaderOptionsBuilder byte_range_offset(self, size_t byte_range_offset)
    cdef CsvReaderOptionsBuilder byte_range_size(self, size_t byte_range_size)
    cdef CsvReaderOptionsBuilder nrows(self, size_type nrows)
    cdef CsvReaderOptionsBuilder skiprows(self, size_type skiprows)
    cdef CsvReaderOptionsBuilder skipfooter(self, size_type skipfooter)
    cdef CsvReaderOptionsBuilder quoting(self, quote_style quoting)
    cdef CsvReaderOptionsBuilder lineterminator(self, str lineterminator)
    cdef CsvReaderOptionsBuilder quotechar(self, str quotechar)
    cdef CsvReaderOptionsBuilder decimal(self, str decimal)
    cdef CsvReaderOptionsBuilder delim_whitespace(self, bool delim_whitespace)
    cdef CsvReaderOptionsBuilder skipinitialspace(self, bool skipinitialspace)
    cdef CsvReaderOptionsBuilder skip_blank_lines(self, bool skip_blank_lines)
    cdef CsvReaderOptionsBuilder doublequote(self, bool doublequote)
    cdef CsvReaderOptionsBuilder keep_default_na(self, bool keep_default_na)
    cdef CsvReaderOptionsBuilder na_filter(self, bool na_filter)
    cdef CsvReaderOptionsBuilder dayfirst(self, bool dayfirst)
    cdef CsvReaderOptionsBuilder build(self)

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
