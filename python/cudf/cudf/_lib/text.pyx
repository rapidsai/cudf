# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from io import TextIOBase

import cudf

from cython.operator cimport dereference
from libc.stdint cimport uint64_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.io.text cimport (
    byte_range_info,
    data_chunk_source,
    make_source,
    make_source_from_bgzip_file,
    make_source_from_file,
    multibyte_split,
)


class BGZIPFile:
    def __init__(self, filename, compression_offsets):
        self.filename = filename
        self.has_offsets = compression_offsets is not None
        if self.has_offsets:
            if len(compression_offsets) != 2:
                raise ValueError(
                    "compression offsets need to consist of two elements")
            self.begin_offset = compression_offsets[0]
            self.end_offset = compression_offsets[1]


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
    cdef uint64_t c_compression_begin_offset
    cdef uint64_t c_compression_end_offset

    if isinstance(filepaths_or_buffers, TextIOBase):
        datasource = move(make_source(filepaths_or_buffers.read().encode()))
    elif isinstance(filepaths_or_buffers, BGZIPFile):
        if filepaths_or_buffers.has_offsets:
            c_compression_begin_offset = filepaths_or_buffers.begin_offset
            c_compression_end_offset = filepaths_or_buffers.end_offset
            datasource = move(make_source_from_bgzip_file(
                filepaths_or_buffers.filename.encode(),
                c_compression_begin_offset,
                c_compression_end_offset))
        else:
            datasource = move(make_source_from_bgzip_file(
                filepaths_or_buffers.filename.encode()))
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
