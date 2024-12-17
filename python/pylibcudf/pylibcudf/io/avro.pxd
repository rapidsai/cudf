# Copyright (c) 2024, NVIDIA CORPORATION.
from pylibcudf.io.types cimport SourceInfo, TableWithMetadata
from pylibcudf.libcudf.io.avro cimport avro_reader_options, avro_reader_options_builder
from pylibcudf.libcudf.types cimport size_type


from pylibcudf.libcudf.types cimport size_type

cdef class AvroReaderOptions:
    cdef avro_reader_options c_obj
    cdef SourceInfo source
    cpdef void set_columns(self, list col_names)


cdef class AvroReaderOptionsBuilder:
    cdef avro_reader_options_builder c_obj
    cdef SourceInfo source
    cpdef AvroReaderOptionsBuilder columns(self, list col_names)
    cpdef AvroReaderOptionsBuilder skip_rows(self, size_type skip_rows)
    cpdef AvroReaderOptionsBuilder num_rows(self, size_type num_rows)
    cpdef AvroReaderOptions build(self)

cpdef TableWithMetadata read_avro(AvroReaderOptions options)
