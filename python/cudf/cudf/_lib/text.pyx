# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from io import TextIOBase

import cudf

from cython.operator cimport dereference
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.io.text cimport (
    byte_range_info,
    data_chunk_source,
    make_source,
    make_source_from_file,
    multibyte_split,
)


def read_text(object filepaths_or_buffers,
              object delimiter=None,
              object byte_range=None):
    """
    Cython function to call into libcudf API, see `multibyte_split`.

    See Also
    --------
    cudf.io.text.read_text
    """
    cdef string delim = delimiter.encode()

    cdef unique_ptr[data_chunk_source] datasource
    cdef unique_ptr[column] c_col

    cdef size_t c_byte_range_offset
    cdef size_t c_byte_range_size
    cdef byte_range_info c_byte_range

    if isinstance(filepaths_or_buffers, TextIOBase):
        datasource = move(make_source(filepaths_or_buffers.read().encode()))
    else:
        datasource = move(make_source_from_file(filepaths_or_buffers.encode()))

    if (byte_range is None):
        with nogil:
            c_col = move(multibyte_split(dereference(datasource), delim))
    else:
        c_byte_range_offset = byte_range[0]
        c_byte_range_size = byte_range[1]
        c_byte_range = byte_range_info(
            c_byte_range_offset,
            c_byte_range_size)
        with nogil:
            c_col = move(multibyte_split(
                dereference(datasource),
                delim,
                c_byte_range))

    return {None: Column.from_unique_ptr(move(c_col))}
