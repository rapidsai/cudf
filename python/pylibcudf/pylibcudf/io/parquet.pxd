# Copyright (c) 2024, NVIDIA CORPORATION.

from libc.stdint cimport int64_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibcudf.expressions cimport Expression
from pylibcudf.io.types cimport SourceInfo, TableWithMetadata
from pylibcudf.libcudf.io.parquet cimport (
    chunked_parquet_reader as cpp_chunked_parquet_reader,
)
from pylibcudf.libcudf.types cimport size_type
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
    bool allow_mismatched_pq_schemas = *,
    int64_t skip_rows = *,
    size_type nrows = *,
    # disabled see comment in parquet.pyx for more
    # ReaderColumnSchema reader_column_schema = *,
    # DataType timestamp_type = *
)
