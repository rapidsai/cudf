# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings.convert cimport convert_urls as cpp_convert_urls
from pylibcudf.utils cimport _get_stream, _get_memory_resource

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

__all__ = ["url_decode", "url_encode"]

cpdef Column url_encode(Column input, Stream stream=None, DeviceMemoryResource mr=None):
    """
    Encodes each string using URL encoding.

    For details, see :cpp:func:`url_encode`

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
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_convert_urls.url_encode(input.view(), stream.view(), mr.get_mr())

    return Column.from_libcudf(move(c_result), stream, mr)


cpdef Column url_decode(Column input, Stream stream=None, DeviceMemoryResource mr=None):
    """
    Decodes each string using URL encoding.

    For details, see :cpp:func:`url_decode`

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
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_convert_urls.url_decode(input.view(), stream.view(), mr.get_mr())

    return Column.from_libcudf(move(c_result), stream, mr)
