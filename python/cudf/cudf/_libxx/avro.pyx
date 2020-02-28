# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.lib cimport table, size_type
from cudf._libxx.table cimport Table
from cudf._libxx.io.utils cimport make_source_info
from cudf._libxx.io.types cimport move, table_with_metadata
from cudf._libxx.io.functions cimport read_avro_args
from cudf._libxx.io.functions cimport read_avro as read_avro_cpp

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

import errno
from io import BytesIO, StringIO
import os


cpdef read_avro(filepath_or_buffer, columns=None, skip_rows=-1, num_rows=-1):
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
        filepath_or_buffer, columns or [], num_rows, skip_rows
    )

    with nogil:
        c_result = move(read_avro_cpp(c_read_avro_args))

    names = [name.decode() for name in c_result.metadata.column_names]

    return Table.from_unique_ptr(move(c_result.tbl), column_names=names)


cdef read_avro_args make_read_avro_args(p, cols, num_rows, skip_rows) except*:
    cdef read_avro_args args = read_avro_args(make_source_info(p))
    args.num_rows = <size_type> num_rows
    args.skip_rows = <size_type> skip_rows
    for col in cols:
        args.columns.push_back(str(col).encode())
    return args
