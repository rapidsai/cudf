# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from io import TextIOBase

from cython.operator cimport dereference
from libc.stdint cimport uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.io.text cimport (
    byte_range_info,
    data_chunk_source,
    make_source,
    make_source_from_bgzip_file,
    make_source_from_file,
    multibyte_split,
    parse_options,
)


def read_text(object filepaths_or_buffers,
              object delimiter=None,
              object byte_range=None,
              object strip_delimiters=False,
              object compression=None,
              object compression_offsets=None):
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
    cdef uint64_t c_compression_begin_offset
    cdef uint64_t c_compression_end_offset
    cdef parse_options c_options

    if compression is None:
        if isinstance(filepaths_or_buffers, TextIOBase):
            datasource = move(make_source(
                filepaths_or_buffers.read().encode()))
        else:
            datasource = move(make_source_from_file(
                filepaths_or_buffers.encode()))
    elif compression == "bgzip":
        if isinstance(filepaths_or_buffers, TextIOBase):
            raise ValueError("bgzip compression requires a file path")
        if compression_offsets is not None:
            if len(compression_offsets) != 2:
                raise ValueError(
                    "compression offsets need to consist of two elements")
            c_compression_begin_offset = compression_offsets[0]
            c_compression_end_offset = compression_offsets[1]
            datasource = move(make_source_from_bgzip_file(
                filepaths_or_buffers.encode(),
                c_compression_begin_offset,
                c_compression_end_offset))
        else:
            datasource = move(make_source_from_bgzip_file(
                filepaths_or_buffers.encode()))
    else:
        raise ValueError("Only bgzip compression is supported at the moment")

    c_options = parse_options()
    if byte_range is not None:
        c_byte_range_offset = byte_range[0]
        c_byte_range_size = byte_range[1]
        c_options.byte_range = byte_range_info(
            c_byte_range_offset,
            c_byte_range_size)
    c_options.strip_delimiters = strip_delimiters
    with nogil:
        c_col = move(multibyte_split(
            dereference(datasource),
            delim,
            c_options))

    return {None: Column.from_unique_ptr(move(c_col))}
