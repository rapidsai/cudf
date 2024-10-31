# Copyright (c) 2024, NVIDIA CORPORATION.

from libc.stdint cimport uint64_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.io cimport text as cpp_text

cdef class ParseOptions:
    """Parsing options for `multibyte_split`"""
    def __init__(
        self,
        *,
        byte_range=None,
        strip_delimiters=False,
    ):
        self.c_options = cpp_text.parse_options()
        if byte_range is not None:
            c_byte_range_offset = byte_range[0]
            c_byte_range_size = byte_range[1]
            self.c_options.byte_range = cpp_text.byte_range_info(
                c_byte_range_offset,
                c_byte_range_size
            )
        self.c_options.strip_delimiters = strip_delimiters


cdef class DataChunkSource:
    """Data source for `multibyte_split`"""

    def __init__(self):
        raise ValueError(
            "This class cannot be instantiated directly. "
            "Use one of the make_source functions instead"
        )

    @staticmethod
    cdef DataChunkSource from_source(data_chunk_source source):
        cdef DataChunkSource datasource = DataChunkSource.__new__(DataChunkSource)
        datasource.c_data_chunk_source = source
        return datasource


cpdef DataChunkSource make_source(str data):
    """
    Creates a data source capable of producing device-buffered views
    of the given string.

    Parameters
    ----------
    data : str
        The host data to be exposed as a data chunk source.

    Returns
    -------
    DataChunkSource
        The data chunk source for the provided host data.
    """
    cdef data_chunk_source c_source
    cdef string c_data = data.encode()

    with nogil:
        c_source = cpp_text.make_source(c_data)

    return DataChunkSource.from_source(c_source)


cpdef DataChunkSource make_source_from_file(str filename):
    """
    Creates a data source capable of producing device-buffered views of the file.

    Parameters
    ----------
    filename : str
        The filename of the file to be exposed as a data chunk source.

    Returns
    -------
    DataChunkSource
        The data chunk source for the provided filename.
    """
    cdef data_chunk_source c_source
    cdef string c_filename = filename.encode()

    with nogil:
        c_source = cpp_text.make_source_from_file(c_filename)

    return DataChunkSource.from_source(c_source)

cpdef DataChunkSource make_source_from_bgzip_file(
    str filename,
    int virtual_begin=None,
    int virtual_end=None,
):
    """
    Creates a data source capable of producing device-buffered views of
    a BGZIP compressed file with virtual record offsets.

    Parameters
    ----------
    filename : str
        The filename of the BGZIP-compressed file to be exposed as a data chunk source.

    virtual_begin : int, default None
        The virtual (Tabix) offset of the first byte to be read. Its upper 48 bits
        describe the offset into the compressed file, its lower 16 bits describe the
        block-local offset.

    virtual_end : int, default None
        The data chunk source for the provided filename.

    Returns
    -------
    DataChunkSource
        The data chunk source for the provided filename.
    """
    cdef data_chunk_source c_source
    cdef string c_filename = filename.encode()

    if virtual_begin is None and virtual_end is None:
        with nogil:
            c_source = cpp_text.make_source_from_bgzip_file(c_filename)
    elif virtual_begin is not None and virtual_end is not None:
        cdef uint64_t c_virtual_begin = virtual_begin
        cdef uint64_t c_virtual_end = c_virtual_end
        with nogil:
            c_source = cpp_text.make_source_from_bgzip_file(
                c_filename,
                virtual_begin,
                c_virtual_end
            )
    else:
        raise ValueError(
            "virtual_begin and virtual_end must both be None or both be int"
        )
    return DataChunkSource.from_source(c_source)

cpdef Column multibyte_split(
    DataChunkSource source,
    str delimiter,
    ParseOptions options=None
):
    """
    Splits the source text into a strings column using a multiple byte delimiter.

    For details, see :cpp:func:`cudf::io::text::multibyte_split`

    Parameters
    ----------
    source :
        The source string.

    delimiter : str
        UTF-8 encoded string for which to find offsets in the source.

    options : ParseOptions
        The parsing options to use (including byte range).

    Returns
    -------
    Column
        The strings found by splitting the source by the delimiter
        within the relevant byte range.
    """
    cdef unique_ptr[column] c_result
    cdef data_chunk_source c_source = source.c_data_chunk_source
    cdef string c_delimiter = delimiter.encode()

    if options is None:
        options = ParseOptions()

    cdef cpp_text.parse_options c_options = options.c_options

    with nogil:
        c_result = cpp_text.multibyte_split(
            c_source
            c_delimiter,
            c_options
        )

    return Column.from_libcudf(move(c_result))
