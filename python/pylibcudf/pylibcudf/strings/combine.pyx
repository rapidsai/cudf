# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from pylibcudf.libcudf.strings cimport combine as cpp_combine
from pylibcudf.scalar cimport Scalar
from pylibcudf.table cimport Table

from cython.operator import dereference
from pylibcudf.libcudf.strings.combine import \
    output_if_empty_list as OutputIfEmptyList  # no-cython-lint
from pylibcudf.libcudf.strings.combine import \
    separator_on_nulls as SeparatorOnNulls  # no-cython-lint

__all__ = [
    "OutputIfEmptyList",
    "SeparatorOnNulls",
    "concatenate",
    "join_list_elements",
    "join_strings",
]

cpdef Column concatenate(
    Table strings_columns,
    ColumnOrScalar separator,
    Scalar narep=None,
    Scalar col_narep=None,
    separator_on_nulls separate_nulls=separator_on_nulls.YES,
):
    """
    Concatenate all columns in the table horizontally into one new string
    delimited by an optional separator string.

    Parameters
    ----------
    strings_columns : Table
        Strings for this operation

    separator : Column or Scalar
        Separator(s) for a given row

    narep : Scalar
        String to replace a null separator for a given row.

    col_narep : Scalar
        String that should be used in place of any null strings found in any column.
        An exception is raised when separator is a Scalar.

    separate_nulls : SeparatorOnNulls
        If YES, then the separator is included for null rows.

    Returns
    -------
    Column
        New column with concatenated results
    """
    cdef unique_ptr[column] c_result
    cdef const string_scalar* c_col_narep
    cdef const string_scalar* c_separator

    if narep is None:
        narep = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode())
        )
    cdef const string_scalar* c_narep = <const string_scalar*>(
        narep.c_obj.get()
    )

    if ColumnOrScalar is Column:
        if col_narep is None:
            col_narep = Scalar.from_libcudf(
                cpp_make_string_scalar("".encode())
            )
        c_col_narep = <const string_scalar*>(
            col_narep.c_obj.get()
        )
        with nogil:
            c_result = move(
                cpp_combine.concatenate(
                    strings_columns.view(),
                    separator.view(),
                    dereference(c_narep),
                    dereference(c_col_narep),
                    separate_nulls
                )
            )
    elif ColumnOrScalar is Scalar:
        if col_narep is not None:
            raise ValueError(
                "col_narep cannot be specified when separator is a Scalar"
            )
        c_separator = <const string_scalar*>(separator.c_obj.get())
        with nogil:
            c_result = move(
                cpp_combine.concatenate(
                    strings_columns.view(),
                    dereference(c_separator),
                    dereference(c_narep),
                    separate_nulls
                )
            )
    else:
        raise ValueError("separator must be a Column or a Scalar")
    return Column.from_libcudf(move(c_result))


cpdef Column join_strings(Column input, Scalar separator, Scalar narep):
    """
    Concatenates all strings in the column into one new string delimited
    by an optional separator string.

    Parameters
    ----------
    input : Column
        List of strings columns to concatenate

    separator : Scalar
        Strings column that provides the separator for a given row

    narep : Scalar
        String to replace any null strings found.

    Returns
    -------
    Column
        New column containing one string
    """
    cdef unique_ptr[column] c_result
    cdef const string_scalar* c_separator = <const string_scalar*>(
        separator.c_obj.get()
    )
    cdef const string_scalar* c_narep = <const string_scalar*>(
        narep.c_obj.get()
    )
    with nogil:
        c_result = move(
            cpp_combine.join_strings(
                input.view(),
                dereference(c_separator),
                dereference(c_narep),
            )
        )

    return Column.from_libcudf(move(c_result))


cpdef Column join_list_elements(
    Column lists_strings_column,
    ColumnOrScalar separator,
    Scalar separator_narep,
    Scalar string_narep,
    separator_on_nulls separate_nulls,
    output_if_empty_list empty_list_policy,
):
    """
    Given a lists column of strings (each row is a list of strings),
    concatenates the strings within each row and returns a single strings
    column result.

    Parameters
    ----------
    lists_strings_column : Column
        Column containing lists of strings to concatenate

    separator : Column or Scalar
        String(s) that should inserted between each string from each row.

    separator_narep : Scalar
        String that should be used to replace a null separator.

    string_narep : Scalar
        String to replace null strings in any non-null list row.
        Ignored if separator is a Scalar.

    separate_nulls : SeparatorOnNulls
        If YES, then the separator is included for null rows
        if `narep` is valid

    empty_list_policy : OutputIfEmptyList
        If set to EMPTY_STRING, any input row that is an empty
        list will result in an empty string. Otherwise, it will
        result in a null.


    Returns
    -------
    Column
        New strings column with concatenated results
    """
    cdef unique_ptr[column] c_result
    cdef const string_scalar* c_separator_narep = <const string_scalar*>(
        separator_narep.c_obj.get()
    )
    cdef const string_scalar* c_string_narep = <const string_scalar*>(
        string_narep.c_obj.get()
    )
    cdef const string_scalar* c_separator

    if ColumnOrScalar is Column:
        with nogil:
            c_result = move(
                cpp_combine.join_list_elements(
                    lists_strings_column.view(),
                    separator.view(),
                    dereference(c_separator_narep),
                    dereference(c_string_narep),
                    separate_nulls,
                    empty_list_policy,
                )
            )
    elif ColumnOrScalar is Scalar:
        c_separator = <const string_scalar*>(separator.c_obj.get())
        with nogil:
            c_result = move(
                cpp_combine.join_list_elements(
                    lists_strings_column.view(),
                    dereference(c_separator),
                    dereference(c_separator_narep),
                    separate_nulls,
                    empty_list_policy,
                )
            )
    else:
        raise ValueError("separator must be a Column or a Scalar")
    return Column.from_libcudf(move(c_result))
