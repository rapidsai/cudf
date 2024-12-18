# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp cimport bool
from pylibcudf.io.types cimport (
    SinkInfo,
    SourceInfo,
    TableWithMetadata,
    compression_type,
)
from pylibcudf.libcudf.io.json cimport (
    json_recovery_mode_t,
    json_writer_options,
    json_writer_options_builder,
)
from pylibcudf.libcudf.types cimport size_type
from rmm._cuda.stream cimport Stream
from pylibcudf.table cimport Table


cpdef TableWithMetadata read_json(
    SourceInfo source_info,
    list dtypes = *,
    compression_type compression = *,
    bool lines = *,
    size_t byte_range_offset = *,
    size_t byte_range_size = *,
    bool keep_quotes = *,
    bool mixed_types_as_string = *,
    bool prune_columns = *,
    json_recovery_mode_t recovery_mode = *,
    dict extra_parameters = *,
    Stream stream = *,
)

cdef class JsonWriterOptions:
    cdef json_writer_options c_obj
    cdef SinkInfo sink
    cdef Table table
    cpdef void set_rows_per_chunk(self, size_type val)
    cpdef void set_true_value(self, str val)
    cpdef void set_false_value(self, str val)

cdef class JsonWriterOptionsBuilder:
    cdef json_writer_options_builder c_obj
    cdef SinkInfo sink
    cdef Table table
    cpdef JsonWriterOptionsBuilder metadata(self, TableWithMetadata tbl_w_meta)
    cpdef JsonWriterOptionsBuilder na_rep(self, str val)
    cpdef JsonWriterOptionsBuilder include_nulls(self, bool val)
    cpdef JsonWriterOptionsBuilder lines(self, bool val)
    cpdef JsonWriterOptions build(self)

cpdef void write_json(JsonWriterOptions options, Stream stream = *)

cpdef tuple chunked_read_json(
    SourceInfo source_info,
    list dtypes = *,
    compression_type compression = *,
    bool keep_quotes = *,
    bool mixed_types_as_string = *,
    bool prune_columns = *,
    json_recovery_mode_t recovery_mode = *,
    int chunk_size= *,
    Stream stream = *,
)
