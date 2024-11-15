# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from cython.operator import dereference

from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from pylibcudf.libcudf.strings cimport contains as cpp_contains
from pylibcudf.strings.regex_program cimport RegexProgram

__all__ = ["contains_re", "count_re", "like", "matches_re"]

cpdef Column contains_re(
    Column input,
    RegexProgram prog
):
    """Returns a boolean column identifying rows which match the given
    regex_program object.

    For details, see :cpp:func:`cudf::strings::contains_re`.

    Parameters
    ----------
    input : Column
        The input strings
    prog : RegexProgram
        Regex program instance

    Returns
    -------
    pylibcudf.Column
        New column of boolean results for each string
    """

    cdef unique_ptr[column] result

    with nogil:
        result = cpp_contains.contains_re(
            input.view(),
            prog.c_obj.get()[0]
        )

    return Column.from_libcudf(move(result))


cpdef Column count_re(
    Column input,
    RegexProgram prog
):
    """Returns the number of times the given regex_program's pattern
    matches in each string.

    For details, see :cpp:func:`cudf::strings::count_re`.

    Parameters
    ----------
    input : Column
        The input strings
    prog : RegexProgram
        Regex program instance

    Returns
    -------
    pylibcudf.Column
        New column of match counts for each string
    """

    cdef unique_ptr[column] result

    with nogil:
        result = cpp_contains.count_re(
            input.view(),
            prog.c_obj.get()[0]
        )

    return Column.from_libcudf(move(result))


cpdef Column matches_re(
    Column input,
    RegexProgram prog
):
    """Returns a boolean column identifying rows which
    matching the given regex_program object but only at
    the beginning the string.

    For details, see :cpp:func:`cudf::strings::matches_re`.

    Parameters
    ----------
    input : Column
        The input strings
    prog : RegexProgram
        Regex program instance

    Returns
    -------
    pylibcudf.Column
        New column of boolean results for each string
    """

    cdef unique_ptr[column] result

    with nogil:
        result = cpp_contains.matches_re(
            input.view(),
            prog.c_obj.get()[0]
        )

    return Column.from_libcudf(move(result))


cpdef Column like(Column input, ColumnOrScalar pattern, Scalar escape_character=None):
    """
    Returns a boolean column identifying rows which
    match the given like pattern.

    For details, see :cpp:func:`cudf::strings::like`.

    Parameters
    ----------
    input : Column
        The input strings
    pattern : Column or Scalar
        Like patterns to match within each string
    escape_character : Scalar
        Optional character specifies the escape prefix.
        Default is no escape character.

    Returns
    -------
    pylibcudf.Column
        New column of boolean results for each string
    """
    cdef unique_ptr[column] result

    if escape_character is None:
        escape_character = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode())
        )

    cdef const string_scalar* c_escape_character = <const string_scalar*>(
        escape_character.c_obj.get()
    )
    cdef const string_scalar* c_pattern

    if ColumnOrScalar is Column:
        with nogil:
            result = cpp_contains.like(
                input.view(),
                pattern.view(),
                dereference(c_escape_character)
            )
    elif ColumnOrScalar is Scalar:
        c_pattern = <const string_scalar*>(pattern.c_obj.get())
        with nogil:
            result = cpp_contains.like(
                input.view(),
                dereference(c_pattern),
                dereference(c_escape_character)
            )
    else:
        raise ValueError("pattern must be a Column or a Scalar")

    return Column.from_libcudf(move(result))
