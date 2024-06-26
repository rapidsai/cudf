# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.vector cimport vector

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

    @staticmethod
    cdef dict _parse_col_names(vector[column_name_info] infos)

    @staticmethod
    cdef TableWithMetadata from_libcudf(table_with_metadata& tbl)

cdef class SourceInfo:
    cdef source_info c_obj
