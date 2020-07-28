# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.io.functions cimport (
    read_avro_args,
    read_avro as libcudf_read_avro
)
from cudf._lib.cpp.io.types cimport table_with_metadata
from cudf._lib.cpp.types cimport size_type
from cudf._lib.io.utils cimport make_source_info
from cudf._lib.move cimport move
from cudf._lib.table cimport Table


cpdef read_avro(datasource, columns=None, skip_rows=-1, num_rows=-1):
    """
    Cython function to call libcudf++ read_avro, see `read_avro`.

    See Also
    --------
    cudf.io.avro.read_avro
    """

    num_rows = -1 if num_rows is None else num_rows
    skip_rows = -1 if skip_rows is None else skip_rows

    if not isinstance(num_rows, int) or num_rows < -1:
        raise TypeError("num_rows must be an int >= -1")
    if not isinstance(skip_rows, int) or skip_rows < -1:
        raise TypeError("skip_rows must be an int >= -1")

    cdef table_with_metadata c_result
    cdef read_avro_args c_read_avro_args = make_read_avro_args(
        datasource, columns or [], num_rows, skip_rows
    )

    with nogil:
        c_result = move(libcudf_read_avro(c_read_avro_args))

    names = [name.decode() for name in c_result.metadata.column_names]

    return Table.from_unique_ptr(move(c_result.tbl), column_names=names)


cdef read_avro_args make_read_avro_args(datasource,
                                        column_names,
                                        num_rows, skip_rows) except*:
    cdef read_avro_args args = read_avro_args(
        make_source_info([datasource])
    )
    args.num_rows = <size_type> num_rows
    args.skip_rows = <size_type> skip_rows
    args.columns.reserve(len(column_names))
    for col in column_names:
        args.columns.push_back(str(col).encode())
    return args
