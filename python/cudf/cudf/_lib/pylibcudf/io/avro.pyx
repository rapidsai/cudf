# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.pylibcudf.io.avro cimport AvroReaderOptions
from cudf._lib.pylibcudf.io.types cimport SourceInfo, TableWithMetadata
from cudf._lib.pylibcudf.libcudf.io.avro cimport (
    avro_reader_options,
    read_avro as cpp_read_avro,
)
from cudf._lib.pylibcudf.libcudf.io.types cimport table_with_metadata
from cudf._lib.pylibcudf.libcudf.types cimport size_type


cdef class AvroReaderOptions:
    def __init__(
        self,
        SourceInfo source_info,
        list columns,
        size_type skip_rows,
        size_type num_rows
    ):
        cdef vector[string] c_columns
        if columns is not None and len(columns) > 0:
            c_columns.reserve(len(columns))
            for col in columns:
                c_columns.push_back(str(col).encode())

        self.avro_opts = move(
            avro_reader_options.builder(source_info.c_obj)
            .columns(c_columns)
            .skip_rows(skip_rows)
            .num_rows(num_rows)
            .build()
        )

cpdef TableWithMetadata read_avro(AvroReaderOptions options):
    """
    Reads an Avro dataset into a set of columns.

    Parameters
    ----------
    options : AvroReaderOptions
        The set of options to pass to the Avro reader.

    Returns
    -------
    TableWithMetadata
        The Table and its corresponding metadata that was read in.
    """
    cdef table_with_metadata c_result

    with nogil:
        c_result = move(cpp_read_avro(options.avro_opts))

    return TableWithMetadata.from_libcudf(c_result)
