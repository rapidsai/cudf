# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport case as cpp_case
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

__all__ = ["swapcase", "to_lower", "to_upper"]

cpdef Column to_lower(Column input, Stream stream=None, DeviceMemoryResource mr=None):
    """Returns a column of lowercased strings.

    For details, see :cpp:func:`to_lower`.

    Parameters
    ----------
    input : Column
        String column
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    pylibcudf.Column
        Column of strings lowercased from the input column
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)
    with nogil:
        c_result = cpp_case.to_lower(input.view(), stream.view(), mr.get_mr())

    return Column.from_libcudf(move(c_result), stream, mr)

cpdef Column to_upper(Column input, Stream stream=None, DeviceMemoryResource mr=None):
    """Returns a column of uppercased strings.

    For details, see :cpp:func:`to_upper`.

    Parameters
    ----------
    input : Column
        String column
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    pylibcudf.Column
        Column of strings uppercased from the input column
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)
    with nogil:
        c_result = cpp_case.to_upper(input.view(), stream.view(), mr.get_mr())

    return Column.from_libcudf(move(c_result), stream, mr)

cpdef Column swapcase(Column input, Stream stream=None, DeviceMemoryResource mr=None):
    """Returns a column of strings where the lowercase characters
    are converted to uppercase and the uppercase characters
    are converted to lowercase.

    For details, see :cpp:func:`swapcase`.

    Parameters
    ----------
    input : Column
        String column
    stream : Stream | None
        CUDA stream on which to perform the operation.
    mr : DeviceMemoryResource | None
        Device memory resource used to allocate the returned column's device memory.

    Returns
    -------
    pylibcudf.Column
        Column of strings
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)
    with nogil:
        c_result = cpp_case.swapcase(input.view(), stream.view(), mr.get_mr())

    return Column.from_libcudf(move(c_result), stream, mr)
