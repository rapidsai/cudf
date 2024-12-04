# Copyright (c) 2024, NVIDIA CORPORATION.

from libc.stdint cimport int64_t, uint8_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from pylibcudf.expressions cimport Expression
from pylibcudf.io.types cimport (
    compression_type,
    dictionary_policy,
    statistics_freq,
    SinkInfo,
    SourceInfo,
    TableInputMetadata,
    TableWithMetadata,
)
from pylibcudf.libcudf.io.parquet cimport (
    chunked_parquet_reader as cpp_chunked_parquet_reader,
    parquet_writer_options,
    parquet_writer_options_builder,
)
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.table cimport Table
from pylibcudf.types cimport DataType


cdef class ChunkedParquetReader:
    cdef unique_ptr[cpp_chunked_parquet_reader] reader

    cpdef bool has_next(self)
    cpdef TableWithMetadata read_chunk(self)


cpdef read_parquet(
    SourceInfo source_info,
    list columns = *,
    list row_groups = *,
    Expression filters = *,
    bool convert_strings_to_categories = *,
    bool use_pandas_metadata = *,
    int64_t skip_rows = *,
    size_type nrows = *,
    bool allow_mismatched_pq_schemas = *,
    # disabled see comment in parquet.pyx for more
    # ReaderColumnSchema reader_column_schema = *,
    # DataType timestamp_type = *
)

cdef class ParquetWriterOptions:
    cdef parquet_writer_options c_obj
    cdef Table table_ref
    cdef SinkInfo sink_ref

    cpdef void set_partitions(self, list partitions)

    cpdef void set_column_chunks_file_paths(self, list file_paths)

    cpdef void set_row_group_size_bytes(self, size_t size_bytes)

    cpdef void set_row_group_size_rows(self, size_type size_rows)

    cpdef void set_max_page_size_bytes(self, size_t size_bytes)

    cpdef void set_max_page_size_rows(self, size_type size_rows)

    cpdef void set_max_dictionary_size(self, size_t size_bytes)

cdef class ParquetWriterOptionsBuilder:
    cdef parquet_writer_options_builder c_obj
    cdef Table table_ref
    cdef SinkInfo sink_ref

    cpdef ParquetWriterOptionsBuilder metadata(self, TableInputMetadata metadata)

    cpdef ParquetWriterOptionsBuilder key_value_metadata(self, list metadata)

    cpdef ParquetWriterOptionsBuilder compression(self, compression_type compression)

    cpdef ParquetWriterOptionsBuilder stats_level(self, statistics_freq sf)

    cpdef ParquetWriterOptionsBuilder int96_timestamps(self, bool enabled)

    cpdef ParquetWriterOptionsBuilder write_v2_headers(self, bool enabled)

    cpdef ParquetWriterOptionsBuilder dictionary_policy(self, dictionary_policy val)

    cpdef ParquetWriterOptionsBuilder utc_timestamps(self, bool enabled)

    cpdef ParquetWriterOptionsBuilder write_arrow_schema(self, bool enabled)

    cpdef ParquetWriterOptions build(self)

cpdef memoryview write_parquet(ParquetWriterOptions options)

cpdef memoryview merge_row_group_metadata(list metdata_list)
