# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.string cimport string

from cudf._lib.pylibcudf.io.types cimport SinkInfo, TableWithMetadata
from cudf._lib.pylibcudf.libcudf.io.types cimport compression_type


cpdef void write_json(
    SinkInfo sink_info,
    TableWithMetadata tbl,
    str na_rep = *,
    bool include_nulls = *,
    bool lines = *,
    int rows_per_chunk = *,
    str true_value = *,
    str false_value = *
)
