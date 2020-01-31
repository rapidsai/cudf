# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from cudf._libxx.lib cimport *
from cudf._libxx.json cimport (
    reader as json_reader,
    reader_options as json_reader_options
)
from cudf._libxx.table cimport *

from libcpp.string cimport string
from libcpp.memory cimport unique_ptr

from cudf._lib.utils cimport *
from cudf._lib.utils import *

import collections.abc as abc
import os


cpdef read_json(filepath_or_buffer, dtype, lines, compression, byte_range):
    """
    Cython function to call into libcudf API, see `read_json`.

    See Also
    --------
    cudf.io.json.read_json
    cudf.io.json.to_json
    """

    # Setup arguments
    cdef reader_options args = reader_options()
    args.lines = lines
    if compression is not None:
        args.compression = compression_type.auto
    else:
        args.compression = compression_type.none
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

    # Create reader from source
    cdef const unsigned char[::1] buffer = view_of_buffer(filepath_or_buffer)
    cdef string filepath
    if buffer is None:
        if os.path.isfile(filepath_or_buffer):
            filepath = <string>str(filepath_or_buffer).encode()
        else:
            buffer = filepath_or_buffer.encode()

    cdef unique_ptr[json_reader] reader
    with nogil:
        if buffer is None:
            reader = unique_ptr[json_reader](
                new json_reader(filepath, args)
            )
        else:
            reader = unique_ptr[json_reader](
                new json_reader(<char *>&buffer[0], buffer.shape[0], args)
            )

    # Read data into columns
    cdef table_with_metadata c_out_table
    cdef size_t c_range_offset = byte_range[0] if byte_range is not None else 0
    cdef size_t c_range_size = byte_range[1] if byte_range is not None else 0
    with nogil:
        if c_range_offset != 0 or c_range_size != 0:
            c_out_table = reader.get().read_byte_range(
                c_range_offset, c_range_size
            )
        else:
            c_out_table = reader.get().read_all()

    column_names = list(c_out_table.metadata.column_names)
    return Table.from_unique_ptr(c_out_table.tbl, column_names=column_names)
