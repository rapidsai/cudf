# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool

from cudf._lib.pylibcudf.io.types cimport SinkInfo, TableWithMetadata
from cudf._lib.pylibcudf.libcudf.types cimport size_type


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
