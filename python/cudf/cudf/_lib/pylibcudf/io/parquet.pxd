# Copyright (c) 2024, NVIDIA CORPORATION.

from libc.stdint cimport int64_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.expressions cimport Expression
from cudf._lib.pylibcudf.io.types cimport SourceInfo, TableWithMetadata
from cudf._lib.pylibcudf.libcudf.io.parquet cimport (
    chunked_parquet_reader as cpp_chunked_parquet_reader,
)
from cudf._lib.pylibcudf.libcudf.types cimport size_type
from cudf._lib.pylibcudf.types cimport DataType


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
    size_type num_rows = *,
    # disabled see comment in parquet.pyx for more
    # ReaderColumnSchema reader_column_schema = *,
    # DataType timestamp_type = *
)
