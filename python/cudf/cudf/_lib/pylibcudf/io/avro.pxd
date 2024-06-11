# Copyright (c) 2024, NVIDIA CORPORATION.
from cudf._lib.pylibcudf.io.types cimport SourceInfo, TableWithMetadata
from cudf._lib.pylibcudf.libcudf.io.avro cimport avro_reader_options
from cudf._lib.pylibcudf.libcudf.types cimport size_type


cpdef TableWithMetadata read_avro(
    SourceInfo source_info,
    list columns = *,
    size_type skip_rows = *,
    size_type num_rows = *
)
