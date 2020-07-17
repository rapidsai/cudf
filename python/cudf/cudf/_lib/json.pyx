# Copyright (c) 2019-2020, NVIDIA CORPORATION.

# cython: boundscheck = False


import cudf
import collections.abc as abc
import io
import os
from cudf._lib.cpp.io.functions cimport (
    read_json as libcudf_read_json,
    read_json_args
)
from cudf._lib.io.utils cimport make_source_info
from cudf._lib.move cimport move
from cudf._lib.table cimport Table
cimport cudf._lib.cpp.io.types as cudf_io_types


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
    path_or_data = filepath_or_buffer

    # If input data is a JSON string (or StringIO), hold a reference to
    # the encoded memoryview externally to ensure the encoded buffer
    # isn't destroyed before calling libcudf++ `read_json()`
    if isinstance(path_or_data, io.StringIO):
        path_or_data = path_or_data.read().encode()
    elif isinstance(path_or_data, str) and not os.path.isfile(path_or_data):
        path_or_data = path_or_data.encode()

    # Setup arguments
    cdef read_json_args args = read_json_args(make_source_info([path_or_data]))

    args.lines = lines
    if compression is not None:
        if compression == 'gzip':
            args.compression = cudf_io_types.compression_type.GZIP
        elif compression == 'bz2':
            args.compression = cudf_io_types.compression_type.BZIP2
        else:
            args.compression = cudf_io_types.compression_type.AUTO
    else:
        args.compression = cudf_io_types.compression_type.NONE

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
    cdef cudf_io_types.table_with_metadata c_out_table

    with nogil:
        c_out_table = move(libcudf_read_json(args))

    column_names = [x.decode() for x in c_out_table.metadata.column_names]
    return Table.from_unique_ptr(move(c_out_table.tbl),
                                 column_names=column_names)
