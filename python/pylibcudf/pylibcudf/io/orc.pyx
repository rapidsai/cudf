# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from pylibcudf.io.types cimport SourceInfo, TableWithMetadata
from pylibcudf.libcudf.io.orc cimport (
    orc_reader_options,
    read_orc as cpp_read_orc,
)
from pylibcudf.libcudf.io.types cimport table_with_metadata
from pylibcudf.libcudf.types cimport size_type, type_id
from pylibcudf.types cimport DataType


cpdef TableWithMetadata read_orc(
    SourceInfo source_info,
    list columns = None,
    list stripes = None,
    size_type skip_rows = 0,
    size_type num_rows = -1,
    bool use_index = True,
    bool use_np_dtypes = True,
    DataType timestamp_type = DataType(type_id.EMPTY),
    list decimal128_columns = None,
):
    """
    """
    cdef orc_reader_options opts
    cdef vector[vector[size_type]] c_stripes
    opts = move(
        orc_reader_options.builder(source_info.c_obj)
        .use_index(use_index)
        .build()
    )
    if num_rows >= 0:
        opts.set_num_rows(num_rows)
    if skip_rows >= 0:
        opts.set_skip_rows(skip_rows)
    if stripes is not None:
        c_stripes = stripes
        opts.set_stripes(c_stripes)
    if timestamp_type.id() != type_id.EMPTY:
        opts.set_timestamp_type(timestamp_type.c_obj)

    cdef vector[string] c_column_names
    if columns is not None:
        c_column_names.reserve(len(columns))
        for col in columns:
            if not isinstance(col, str):
                raise TypeError("Column names must be strings!")
            c_column_names.push_back(str(col).encode())
        if len(columns) > 0:
            opts.set_columns(c_column_names)

    cdef table_with_metadata c_result

    with nogil:
        c_result = move(cpp_read_orc(opts))

    return TableWithMetadata.from_libcudf(c_result)
