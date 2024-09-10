# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings cimport combine as cpp_combine
from pylibcudf.scalar cimport Scalar
from pylibcudf.table cimport Table

from pylibcudf.libcudf.strings.combine import \
    output_if_empty_list as OutputIfEmptyList  # no-cython-lint
from pylibcudf.libcudf.strings.combine import \
    separator_on_nulls as SeparatorOnNulls  # no-cython-lint


cpdef Column concatenate(
    Table strings_columns,
    ColumnOrScalar separator,
    Scalar narep,
    Scalar col_narep,
    separator_on_nulls separate_nulls,
):
    """
    Concatenates all strings in the column into one new string delimited
    by an optional separator string.

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
        Ignored when separator is a Scalar.

    separate_nulls : SeparatorOnNulls
        If YES, then the separator is included for null rows.

    Returns
    -------
    Column
        New column with concatenated results
    """
    cdef unique_ptr[column] c_result
    cdef const string_scalar* c_narep = <const string_scalar*>(
        narep.c_obj.get()
    )
    cdef const string_scalar* c_col_narep = <const string_scalar*>(
        narep.c_obj.get()
    )
    if ColumnOrScalar is Column:
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
        cdef const string_scalar* c_separator = <const string_scalar*>(
            separator.c_obj.get()
        )
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
                c_separator,
                c_narep,
            )
        )

    return Column.from_libcudf(move(c_result))
