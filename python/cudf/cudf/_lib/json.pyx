# Copyright (c) 2019-2020, NVIDIA CORPORATION.

# cython: boundscheck = False


import cudf
import collections.abc as abc
import io
import os

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport move

from cudf._lib.cpp.io.json cimport (
    read_json as libcudf_read_json,
    json_reader_options
)
from cudf._lib.cpp.types cimport size_type
from cudf._lib.io.utils cimport make_source_info
from cudf._lib.table cimport Table
cimport cudf._lib.cpp.io.types as cudf_io_types


cpdef read_json(object filepath_or_buffer,
                object dtype,
                bool lines,
                object compression,
                object byte_range):
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
    cdef vector[string] c_dtypes
    cdef cudf_io_types.compression_type c_compression
    # Determine byte read offsets if applicable
    cdef size_type c_range_offset = (
        byte_range[0] if byte_range is not None else 0
    )
    cdef size_type c_range_size = (
        byte_range[1] if byte_range is not None else 0
    )
    cdef bool c_lines = lines

    if compression is not None:
        if compression == 'gzip':
            c_compression = cudf_io_types.compression_type.GZIP
        elif compression == 'bz2':
            c_compression = cudf_io_types.compression_type.BZIP2
        else:
            c_compression = cudf_io_types.compression_type.AUTO
    else:
        c_compression = cudf_io_types.compression_type.NONE

    if dtype is False:
        raise ValueError("False value is unsupported for `dtype`")
    elif dtype is not True:
        if isinstance(dtype, abc.Mapping):
            c_dtypes.reserve(len(dtype))
            for k, v in dtype.items():
                c_dtypes.push_back(str(str(k) + ":" + str(v)).encode())
        elif not isinstance(dtype, abc.Iterable):
            raise TypeError("`dtype` must be 'list like' or 'dict'")
        else:
            c_dtypes.reserve(len(dtype))
            for col_dtype in dtype:
                c_dtypes.push_back(str(col_dtype).encode())

    cdef json_reader_options opts = move(
        json_reader_options.builder(make_source_info([path_or_data]))
        .dtypes(c_dtypes)
        .compression(c_compression)
        .lines(c_lines)
        .byte_range_offset(c_range_offset)
        .byte_range_size(c_range_size)
        .build()
    )

    # Read JSON
    cdef cudf_io_types.table_with_metadata c_out_table

    with nogil:
        c_out_table = move(libcudf_read_json(opts))

    column_names = [x.decode() for x in c_out_table.metadata.column_names]
    return Table.from_unique_ptr(move(c_out_table.tbl),
                                 column_names=column_names)
