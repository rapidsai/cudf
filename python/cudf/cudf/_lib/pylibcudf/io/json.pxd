# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp cimport bool

from cudf._lib.pylibcudf.io.types cimport (
    SinkInfo,
    SourceInfo,
    TableWithMetadata,
    compression_type,
)
from cudf._lib.pylibcudf.libcudf.io.json cimport json_recovery_mode_t
from cudf._lib.pylibcudf.libcudf.types cimport size_type


cpdef TableWithMetadata read_json(
    SourceInfo source_info,
    list dtypes = *,
    compression_type compression = *,
    bool lines = *,
    size_type byte_range_offset = *,
    size_type byte_range_size = *,
    bool keep_quotes = *,
    bool mixed_types_as_string = *,
    bool prune_columns = *,
    json_recovery_mode_t recovery_mode = *,
)


cpdef void write_json(
    SinkInfo sink_info,
    TableWithMetadata tbl,
    str na_rep = *,
    bool include_nulls = *,
    bool lines = *,
    size_type rows_per_chunk = *,
    str true_value = *,
    str false_value = *
)

cpdef tuple chunked_read_json(
    SourceInfo source_info,
    list dtypes = *,
    compression_type compression = *,
    bool keep_quotes = *,
    bool mixed_types_as_string = *,
    bool prune_columns = *,
    json_recovery_mode_t recovery_mode = *,
    int chunk_size= *,
)
