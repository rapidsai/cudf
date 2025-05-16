# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.strings.convert cimport (
    convert_booleans as cpp_convert_booleans,
)
from pylibcudf.scalar cimport Scalar

from cython.operator import dereference

__all__ = ["from_booleans", "to_booleans"]

cpdef Column to_booleans(Column input, Scalar true_string):
    """
    Returns a new bool column by parsing boolean values from the strings
    in the provided strings column.

    For details, see :cpp:func:`cudf::strings::to_booleans`.

    Parameters
    ----------
    input :  Column
        Strings instance for this operation

    true_string : Scalar
        String to expect for true. Non-matching strings are false

    Returns
    -------
    Column
        New bool column converted from strings.
    """
    cdef unique_ptr[column] c_result
    cdef const string_scalar* c_true_string = <const string_scalar*>(
        true_string.c_obj.get()
    )

    with nogil:
        c_result = cpp_convert_booleans.to_booleans(
            input.view(),
            dereference(c_true_string)
        )

    return Column.from_libcudf(move(c_result))

cpdef Column from_booleans(Column booleans, Scalar true_string, Scalar false_string):
    """
    Returns a new strings column converting the boolean values from the
    provided column into strings.

    For details, see :cpp:func:`cudf::strings::from_booleans`.

    Parameters
    ----------
    booleans :  Column
        Boolean column to convert.

    true_string : Scalar
        String to use for true in the output column.

    false_string : Scalar
        String to use for false in the output column.

    Returns
    -------
    Column
        New strings column.
    """
    cdef unique_ptr[column] c_result
    cdef const string_scalar* c_true_string = <const string_scalar*>(
        true_string.c_obj.get()
    )
    cdef const string_scalar* c_false_string = <const string_scalar*>(
        false_string.c_obj.get()
    )

    with nogil:
        c_result = cpp_convert_booleans.from_booleans(
            booleans.view(),
            dereference(c_true_string),
            dereference(c_false_string),
        )

    return Column.from_libcudf(move(c_result))
