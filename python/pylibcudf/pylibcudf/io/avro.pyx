# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.io.types cimport SourceInfo, TableWithMetadata
from pylibcudf.libcudf.io.avro cimport (
    avro_reader_options,
    read_avro as cpp_read_avro,
)
from pylibcudf.libcudf.types cimport size_type

__all__ = ["read_avro"]


cpdef TableWithMetadata read_avro(
    SourceInfo source_info,
    list columns = None,
    size_type skip_rows = 0,
    size_type num_rows = -1
):
    """
    Reads an Avro dataset into a :py:class:`~.types.TableWithMetadata`.

    For details, see :cpp:func:`read_avro`.

    Parameters
    ----------
    source_info: SourceInfo
        The SourceInfo object to read the avro dataset from.
    columns: list, default None
        Optional columns to read, if not provided, reads all columns in the file.
    skip_rows: size_type, default 0
        The number of rows to skip.
    num_rows: size_type, default -1
        The number of rows to read, after skipping rows.
        If -1 is passed, all rows will be read.

    Returns
    -------
    TableWithMetadata
        The Table and its corresponding metadata (column names) that were read in.
    """
    cdef vector[string] c_columns
    if columns is not None and len(columns) > 0:
        c_columns.reserve(len(columns))
        for col in columns:
            c_columns.push_back(str(col).encode())

    cdef avro_reader_options avro_opts = (
        avro_reader_options.builder(source_info.c_obj)
        .columns(c_columns)
        .skip_rows(skip_rows)
        .num_rows(num_rows)
        .build()
    )

    with nogil:
        c_result = move(cpp_read_avro(avro_opts))

    return TableWithMetadata.from_libcudf(c_result)
