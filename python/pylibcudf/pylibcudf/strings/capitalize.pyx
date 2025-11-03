# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from pylibcudf.libcudf.strings cimport capitalize as cpp_capitalize
from pylibcudf.scalar cimport Scalar
from pylibcudf.strings.char_types cimport string_character_types
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from cython.operator import dereference

__all__ = ["capitalize", "is_title", "title"]

cpdef Column capitalize(
    Column input,
    Scalar delimiters=None,
    Stream stream=None,
    DeviceMemoryResource mr=None,
    # TODO: default scalar values
    # https://github.com/rapidsai/cudf/issues/15505
):
    """Returns a column of capitalized strings.

    For details, see :cpp:func:`capitalize`.

    Parameters
    ----------
    input : Column
        String column
    delimiters : Scalar, default None
        Characters for identifying words to capitalize

    Returns
    -------
    pylibcudf.Column
        Column of strings capitalized from the input column
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)

    if delimiters is None:
        delimiters = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode(), stream.view(), mr.get_mr())
        )

    cdef const string_scalar* cpp_delimiters = <const string_scalar*>(
        delimiters.c_obj.get()
    )

    with nogil:
        c_result = cpp_capitalize.capitalize(
            input.view(),
            dereference(cpp_delimiters),
            stream.view(),
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), stream, mr)


cpdef Column title(
    Column input,
    string_character_types sequence_type=string_character_types.ALPHA,
    Stream stream=None,
    DeviceMemoryResource mr=None,
):
    """Modifies first character of each word to upper-case and lower-cases
    the rest.

    For details, see :cpp:func:`title`.

    Parameters
    ----------
    input : Column
        String column
    sequence_type : string_character_types, default string_character_types.ALPHA
        The character type that is used when identifying words

    Returns
    -------
    pylibcudf.Column
        Column of titled strings
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)
    with nogil:
        c_result = cpp_capitalize.title(
            input.view(), sequence_type, stream.view(), mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), stream, mr)


cpdef Column is_title(Column input, Stream stream=None, DeviceMemoryResource mr=None):
    """Checks if the strings in the input column are title formatted.

    For details, see :cpp:func:`is_title`.

    Parameters
    ----------
    input : Column
        String column

    Returns
    -------
    pylibcudf.Column
        Column of type BOOL8
    """
    cdef unique_ptr[column] c_result
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)
    with nogil:
        c_result = cpp_capitalize.is_title(input.view(), stream.view(), mr.get_mr())

    return Column.from_libcudf(move(c_result), stream, mr)
