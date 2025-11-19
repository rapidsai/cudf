# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings.convert cimport (
    convert_integers as cpp_convert_integers,
)
from pylibcudf.types cimport DataType
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

__all__ = [
    "from_integers",
    "hex_to_integers",
    "integers_to_hex",
    "is_hex",
    "is_integer",
    "to_integers"
]

cpdef Column to_integers(
    Column input, DataType output_type, Stream stream=None, DeviceMemoryResource mr=None
):
    """
    Returns a new integer numeric column parsing integer values from the
    provided strings column.

    For details, cpp:func:`cudf::strings::to_integers`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    output_type : DataType
        Type of integer numeric column to return.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column with integers converted from strings.
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = move(
            cpp_convert_integers.to_integers(
                input.view(),
                output_type.c_obj,
                stream.view(),
                mr.get_mr()
            )
        )

    return Column.from_libcudf(move(c_result), stream, mr)


cpdef Column from_integers(
    Column integers, Stream stream=None, DeviceMemoryResource mr=None
):
    """
    Returns a new strings column converting the integer values from the
    provided column into strings.

    For details, cpp:func:`cudf::strings::from_integers`.

    Parameters
    ----------
    integers : Column
        Strings instance for this operation.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings column with integers as strings.
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = move(
            cpp_convert_integers.from_integers(
                integers.view(),
                stream.view(),
                mr.get_mr()
            )
        )

    return Column.from_libcudf(move(c_result), stream, mr)


cpdef Column is_integer(
    Column input,
    DataType int_type=None,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """
    Returns a boolean column identifying strings in which all
    characters are valid for conversion to integers.

    For details, cpp:func:`cudf::strings::is_integer`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    int_type : DataType
        Integer type used for checking underflow and overflow.
        By default, does not check an integer type for underflow
        or overflow.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of boolean results for each string.
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    if int_type is None:
        with nogil:
            c_result = move(
                cpp_convert_integers.is_integer(
                    input.view(),
                    stream.view(),
                    mr.get_mr()
                )
            )
    else:
        with nogil:
            c_result = move(
                cpp_convert_integers.is_integer(
                    input.view(),
                    int_type.c_obj,
                    stream.view(),
                    mr.get_mr()
                )
            )

    return Column.from_libcudf(move(c_result), stream, mr)


cpdef Column hex_to_integers(
    Column input, DataType output_type, Stream stream=None, DeviceMemoryResource mr=None
):
    """
    Returns a new integer numeric column parsing hexadecimal values
    from the provided strings column.

    For details, cpp:func:`cudf::strings::hex_to_integers`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    output_type : DataType
        Type of integer numeric column to return.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column with integers converted from strings.
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = move(
            cpp_convert_integers.hex_to_integers(
                input.view(),
                output_type.c_obj,
                stream.view(),
                mr.get_mr()
            )
        )

    return Column.from_libcudf(move(c_result), stream, mr)


cpdef Column is_hex(Column input, Stream stream=None, DeviceMemoryResource mr=None):
    """
    Returns a boolean column identifying strings in which all
    characters are valid for conversion to integers from hex.

    For details, cpp:func:`cudf::strings::is_hex`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New column of boolean results for each string.
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = move(
            cpp_convert_integers.is_hex(
                input.view(),
                stream.view(),
                mr.get_mr()
            )
        )

    return Column.from_libcudf(move(c_result), stream, mr)


cpdef Column integers_to_hex(
    Column input, Stream stream=None, DeviceMemoryResource mr=None
):
    """
    Returns a new strings column converting integer columns to hexadecimal
    characters.

    For details, cpp:func:`cudf::strings::integers_to_hex`.

    Parameters
    ----------
    input : Column
        Integer column to convert to hex.

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings column with hexadecimal characters.
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    with nogil:
        c_result = move(
            cpp_convert_integers.integers_to_hex(
                input.view(),
                stream.view(),
                mr.get_mr()
            )
        )

    return Column.from_libcudf(move(c_result), stream, mr)
