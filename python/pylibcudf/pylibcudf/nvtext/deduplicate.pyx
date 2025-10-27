# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cython.operator import dereference

from libcpp.memory cimport unique_ptr, make_unique
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.nvtext.deduplicate cimport (
    build_suffix_array as cpp_build_suffix_array,
    suffix_array_type as cpp_suffix_array_type,
    resolve_duplicates as cpp_resolve_duplicates,
    resolve_duplicates_pair as cpp_resolve_duplicates_pair,
)
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.stream cimport Stream

__all__ = [
    "build_suffix_array",
    "resolve_duplicates",
    "resolve_duplicates_pair",
]

cdef Column _column_from_suffix_array(
    cpp_suffix_array_type suffix_array, Stream stream, DeviceMemoryResource mr
):
    # helper to convert a suffix array to a Column
    return Column.from_libcudf(
        move(
            make_unique[column](
                move(dereference(suffix_array.get())),
                device_buffer(),
                0
            )
        ),
        stream,
        mr
    )


cpdef Column build_suffix_array(
    Column input, size_type min_width, Stream stream=None, DeviceMemoryResource mr=None
):
    """
    Builds a suffix array for the input strings column.
    A suffix array is the indices of the sorted set of substrings
    of the input column as: [ input[0:], input[1:], ... input[bytes-1:] ]
    where bytes is the total number of bytes in input.
    The returned array represent the sorted strings such that
    result[i] = input[result[i]:]

    For details, see :cpp:func:`build_suffix_array`

    Parameters
    ----------
    input : Column
        Strings column of text
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of suffix array
    """
    cdef cpp_suffix_array_type c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_build_suffix_array(
            input.view(), min_width, stream.view(), mr.get_mr()
        )

    return _column_from_suffix_array(move(c_result), stream, mr)


cpdef Column resolve_duplicates(
    Column input,
    Column indices,
    size_type min_width,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns duplicate strings found in the input column
    with min_width minimum number of bytes.
    The indices are expected to be the suffix array previously created
    for input. Otherwise, the results are undefined.

    For details, see :cpp:func:`resolve_duplicates`

    Parameters
    ----------
    input : Column
        Strings column of text
    indices : Column
        Suffix array from :cpp:func:`build_suffix_array`
    min_width : size_type
        Minimum width of bytes to detect duplicates
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of duplicate strings
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_resolve_duplicates(
            input.view(), indices.view(), min_width, stream.view(), mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), stream, mr)


cpdef Column resolve_duplicates_pair(
    Column input1,
    Column indices1,
    Column input2,
    Column indices2,
    size_type min_width,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns duplicate strings in input1 found in input2
    with min_width minimum number of bytes.
    The indices are expected to be the suffix array previously created
    for the inputs. Otherwise, the results are undefined.

    For details, see :cpp:func:`resolve_duplicates_pair`

    Parameters
    ----------
    input1 : Column
        Strings column of text
    indices1 : Column
        Suffix array from :cpp:func:`build_suffix_array` for input1
    input2 : Column
        Strings column of text
    indices2 : Column
        Suffix array from :cpp:func:`build_suffix_array` for input2
    min_width : size_type
        Minimum width of bytes to detect duplicates
    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of duplicate strings

    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = cpp_resolve_duplicates_pair(
            input1.view(),
            indices1.view(),
            input2.view(),
            indices2.view(),
            min_width,
            stream.view(),
            mr.get_mr(),
        )

    return Column.from_libcudf(move(c_result), stream, mr)
