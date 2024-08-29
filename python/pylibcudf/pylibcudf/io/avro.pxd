# Copyright (c) 2024, NVIDIA CORPORATION.
from pylibcudf.io.types cimport SourceInfo, TableWithMetadata
from pylibcudf.libcudf.io.avro cimport avro_reader_options
from pylibcudf.libcudf.types cimport size_type


cpdef TableWithMetadata read_avro(
    SourceInfo source_info,
    list columns = *,
    size_type skip_rows = *,
    size_type num_rows = *
)
