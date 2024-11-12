# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings.convert cimport convert_urls as cpp_convert_urls

__all__ = ["url_decode", "url_encode"]

cpdef Column url_encode(Column input):
    """
    Encodes each string using URL encoding.

    For details, see :cpp:func:`cudf::strings::url_encode`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    Returns
    -------
    Column
        New strings column.
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_convert_urls.url_encode(input.view())

    return Column.from_libcudf(move(c_result))


cpdef Column url_decode(Column input):
    """
    Decodes each string using URL encoding.

    For details, see :cpp:func:`cudf::strings::url_decode`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    Returns
    -------
    Column
        New strings column.
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_convert_urls.url_decode(input.view())

    return Column.from_libcudf(move(c_result))
