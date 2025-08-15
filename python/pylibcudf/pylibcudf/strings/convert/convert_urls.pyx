# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings.convert cimport convert_urls as cpp_convert_urls
from pylibcudf.utils cimport _get_stream

from rmm.pylibrmm.stream cimport Stream

__all__ = ["url_decode", "url_encode"]

cpdef Column url_encode(Column input, Stream stream=None):
    """
    Encodes each string using URL encoding.

    For details, see :cpp:func:`cudf::strings::url_encode`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings column.
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_convert_urls.url_encode(input.view(), stream.view())

    return Column.from_libcudf(move(c_result), stream)


cpdef Column url_decode(Column input, Stream stream=None):
    """
    Decodes each string using URL encoding.

    For details, see :cpp:func:`cudf::strings::url_decode`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings column.
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)

    with nogil:
        c_result = cpp_convert_urls.url_decode(input.view(), stream.view())

    return Column.from_libcudf(move(c_result), stream)
