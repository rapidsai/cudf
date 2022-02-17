# Copyright (c) 2020-2022, NVIDIA CORPORATION.

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
    cdef string filename = filepaths_or_buffers.encode()
    cdef string delim = delimiter.encode()

    cdef unique_ptr[data_chunk_source] datasource
    cdef unique_ptr[column] c_col
    cdef size_t c_byte_range_offset
    cdef size_t c_byte_range_size
    cdef byte_range_info c_byte_range

    if (byte_range is not None):
        c_byte_range_offset = byte_range[0]
        c_byte_range_size = byte_range[1]
        with nogil:
            datasource = move(make_source_from_file(filename))
            c_byte_range = byte_range_info(
                c_byte_range_offset,
                c_byte_range_size
            )
            c_col = move(
                multibyte_split(dereference(datasource), delim, c_byte_range)
            )
    else:
        with nogil:
            datasource = move(make_source_from_file(filename))
            c_col = move(multibyte_split(dereference(datasource), delim))

    return {None: Column.from_unique_ptr(move(c_col))}
