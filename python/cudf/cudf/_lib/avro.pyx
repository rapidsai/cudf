# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.io.utils cimport make_source_info
from cudf._lib.pylibcudf.libcudf.io.avro cimport (
    avro_reader_options,
    read_avro as libcudf_read_avro,
)
from cudf._lib.pylibcudf.libcudf.io.types cimport table_with_metadata
from cudf._lib.pylibcudf.libcudf.types cimport size_type
from cudf._lib.utils cimport data_from_unique_ptr


cpdef read_avro(datasource, columns=None, skip_rows=-1, num_rows=-1):
    """
    Cython function to call libcudf read_avro, see `read_avro`.

    See Also
    --------
    cudf.io.avro.read_avro
    """

    num_rows = -1 if num_rows is None else num_rows
    skip_rows = 0 if skip_rows is None else skip_rows

    if not isinstance(num_rows, int) or num_rows < -1:
        raise TypeError("num_rows must be an int >= -1")
    if not isinstance(skip_rows, int) or skip_rows < -1:
        raise TypeError("skip_rows must be an int >= -1")

    cdef vector[string] c_columns
    if columns is not None and len(columns) > 0:
        c_columns.reserve(len(columns))
        for col in columns:
            c_columns.push_back(str(col).encode())

    cdef avro_reader_options options = move(
        avro_reader_options.builder(make_source_info([datasource]))
        .columns(c_columns)
        .skip_rows(<size_type> skip_rows)
        .num_rows(<size_type> num_rows)
        .build()
    )

    cdef table_with_metadata c_result

    with nogil:
        c_result = move(libcudf_read_avro(options))

    names = [info.name.decode() for info in c_result.metadata.schema_info]

    return data_from_unique_ptr(move(c_result.tbl), column_names=names)
