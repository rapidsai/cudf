# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings cimport findall as cpp_findall
from pylibcudf.strings.regex_program cimport RegexProgram
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

__all__ = ["findall", "find_re"]

cpdef Column findall(
    Column input, RegexProgram pattern, Stream stream=None, DeviceMemoryResource mr=None
):
    """
    Returns a lists column of strings for each matching occurrence using
    the regex_program pattern within each string.

    For details, see :cpp:func:`findall`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    pattern : RegexProgram
        Regex pattern
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New lists column of strings
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_findall.findall(
            input.view(),
            pattern.c_obj.get()[0],
            stream.view(),
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), stream, mr)


cpdef Column find_re(
    Column input, RegexProgram pattern, Stream stream=None, DeviceMemoryResource mr=None
):
    """
    Returns character positions where the pattern first matches
    the elements in input strings.

    For details, see :cpp:func:`find_re`

    Parameters
    ----------
    input : Column
        Strings instance for this operation
    pattern : RegexProgram
        Regex pattern
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of integers
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_findall.find_re(
            input.view(),
            pattern.c_obj.get()[0],
            stream.view(),
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), stream, mr)
