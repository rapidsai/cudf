# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from pylibcudf.libcudf.io.csv cimport (
    csv_writer_options,
    csv_writer_options_builder,
)
from pylibcudf.libcudf.io.types cimport quote_style
from pylibcudf.io.types cimport SinkInfo
from pylibcudf.table cimport Table

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
