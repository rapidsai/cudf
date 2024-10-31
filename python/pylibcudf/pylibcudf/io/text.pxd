# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.io.text cimport parse_options,

cdef class ParseOptions:
    cdef parse_options c_options

cdef class DataChunkSource:
    cdef data_chunk_source c_data_chunk_source

    cdef DataChunkSource from_source(data_chunk_source source)


cpdef Column multibyte_split(
    source,
    str delimiter,
    ParseOptions options=*
)

cpdef DataChunkSource make_source(str data)

cpdef DataChunkSource make_source_from_file(str filename)

cpdef DataChunkSource make_source_from_bgzip_file(
    str filename,
    int virtual_begin=*,
    int virtual_end=*,
)
