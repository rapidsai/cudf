# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from cudf._libxx.lib cimport *
from cudf._libxx.table cimport *
from cudf._libxx.io.functions cimport *
from cudf._libxx.io.types cimport *

from libcpp.string cimport string
from libcpp.memory cimport unique_ptr

from cudf._lib.utils cimport *
from cudf._lib.utils import *

import collections.abc as abc
import os


cpdef read_json_libcudf(filepath_or_buffer, dtype,
                        lines, compression, byte_range):
    """
    Cython function to call into libcudf API, see `read_json`.

    See Also
    --------
    cudf.io.json.read_json
    cudf.io.json.to_json
    """

    # Determine read source
    cdef const unsigned char[::1] buffer = view_of_buffer(filepath_or_buffer)
    cdef string filepath
    cdef source_info source
    if buffer is None:
        if os.path.isfile(filepath_or_buffer):
            source.type = io_type.FILEPATH
            filepath = <string>str(filepath_or_buffer).encode()
            source.filepath = filepath
        else:
            source.type = io_type.HOST_BUFFER
            source.filepath = filepath_or_buffer.encode()

    # Setup arguments
    cdef read_json_args args = read_json_args(source)

    args.lines = lines
    if compression is not None:
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
        c_out_table = read_json(args)

    column_names = list(c_out_table.metadata.column_names)
    return Table.from_unique_ptr(c_out_table.tbl, column_names=column_names)
