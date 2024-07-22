# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.libcudf.io.data_sink cimport data_sink
from cudf._lib.pylibcudf.libcudf.io.types cimport (
    column_encoding,
    column_in_metadata,
    column_name_info,
    compression_type,
    dictionary_policy,
    io_type,
    partition_info,
    quote_style,
    sink_info,
    source_info,
    statistics_freq,
    table_input_metadata,
    table_metadata,
    table_with_metadata,
)
from cudf._lib.pylibcudf.table cimport Table


cdef class TableWithMetadata:
    cdef public Table tbl
    cdef table_metadata metadata

    cdef vector[column_name_info] _make_column_info(self, list column_names)

    cdef list _make_columns_list(self, dict child_dict)

    @staticmethod
    cdef dict _parse_col_names(vector[column_name_info] infos)

    @staticmethod
    cdef TableWithMetadata from_libcudf(table_with_metadata& tbl)

cdef class SourceInfo:
    cdef source_info c_obj
    # Keep the bytes converted from stringio alive
    # (otherwise we end up with a use after free when they get gc'ed)
    cdef list byte_sources

cdef class SinkInfo:
    # This vector just exists to keep the unique_ptrs to the sinks alive
    cdef vector[unique_ptr[data_sink]] sink_storage
    cdef sink_info c_obj
