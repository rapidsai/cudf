# Copyright (c) 2019-2020, NVIDIA CORPORATION.

# cython: boundscheck = False


import cudf
from cudf._libxx.lib cimport *
from cudf._libxx.table cimport *
from cudf._libxx.io.types cimport *
from cudf._libxx.io.functions cimport (
    read_json as cpp_read_json,
    read_json_args
)

from libcpp.string cimport string
from libcpp.memory cimport unique_ptr

from cudf._lib.utils cimport *
from cudf._lib.utils import *

import collections.abc as abc
import os


cpdef read_json(filepath_or_buffer, dtype,
                lines, compression, byte_range):
    """
    Cython function to call into libcudf API, see `read_json`.

    See Also
    --------
    cudf.io.json.read_json
    cudf.io.json.to_json
    """

    # Determine read source
    cdef source_info source
    cdef const unsigned char[::1] buffer = view_of_buffer(filepath_or_buffer)
    cdef string filepath
    if buffer is None:
        if os.path.isfile(filepath_or_buffer):
            filepath = <string>str(filepath_or_buffer).encode()
        else:
            buffer = filepath_or_buffer.encode()

    if buffer is None:
        source.type = io_type.FILEPATH
        source.filepath = filepath
    else:
        source.type = io_type.HOST_BUFFER
        source.buffer.first = <char*>&buffer[0]
        source.buffer.second = buffer.shape[0]

    # Setup arguments
    cdef read_json_args args = read_json_args(source)

    args.lines = lines
    if compression is not None:
        if compression == 'gzip':
            args.compression = compression_type.GZIP
        elif compression == 'bz2':
            args.compression = compression_type.BZIP2
        else:
            args.compression = compression_type.AUTO
    else:
        args.compression = compression_type.NONE

    if dtype is False:
        raise ValueError("False value is unsupported for `dtype`")
    elif dtype is not True:
        if isinstance(dtype, abc.Mapping):
            for k, v in dtype.items():
                args.dtype.push_back(str(str(k) + ":" + str(v)).encode())
        elif not isinstance(dtype, abc.Iterable):
            raise TypeError("`dtype` must be 'list like' or 'dict'")
        else:
            for col_dtype in dtype:
                args.dtype.push_back(str(col_dtype).encode())

    # Determine byte read offsets if applicable
    cdef size_t c_range_offset = byte_range[0] if byte_range is not None else 0
    cdef size_t c_range_size = byte_range[1] if byte_range is not None else 0
    args.byte_range_offset = c_range_offset
    args.byte_range_size = c_range_size

    # Read JSON
    cdef table_with_metadata c_out_table

    with nogil:
        c_out_table = move(cpp_read_json(args))

    column_names = list(c_out_table.metadata.column_names)
    column_names = [x.decode() for x in column_names]
    tbl = Table.from_unique_ptr(move(c_out_table.tbl),
                                column_names=column_names)
    return cudf.DataFrame._from_table(tbl)
