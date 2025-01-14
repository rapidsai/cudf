# Copyright (c) 2024, NVIDIA CORPORATION.

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

from cython.operator import dereference

__all__ = ["capitalize", "is_title", "title"]

cpdef Column capitalize(
    Column input,
    Scalar delimiters=None
    # TODO: default scalar values
    # https://github.com/rapidsai/cudf/issues/15505
):
    """Returns a column of capitalized strings.

    For details, see :cpp:func:`cudf::strings::capitalize`.

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

    if delimiters is None:
        delimiters = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode())
        )

    cdef const string_scalar* cpp_delimiters = <const string_scalar*>(
        delimiters.c_obj.get()
    )

    with nogil:
        c_result = cpp_capitalize.capitalize(
            input.view(),
            dereference(cpp_delimiters)
        )

    return Column.from_libcudf(move(c_result))


cpdef Column title(
    Column input,
    string_character_types sequence_type=string_character_types.ALPHA
):
    """Modifies first character of each word to upper-case and lower-cases
    the rest.

    For details, see :cpp:func:`cudf::strings::title`.

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
    with nogil:
        c_result = cpp_capitalize.title(input.view(), sequence_type)

    return Column.from_libcudf(move(c_result))


cpdef Column is_title(Column input):
    """Checks if the strings in the input column are title formatted.

    For details, see :cpp:func:`cudf::strings::is_title`.

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
    with nogil:
        c_result = cpp_capitalize.is_title(input.view())

    return Column.from_libcudf(move(c_result))
