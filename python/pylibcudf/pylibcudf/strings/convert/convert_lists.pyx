# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.column_factories cimport make_empty_column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.scalar.scalar cimport string_scalar
from pylibcudf.libcudf.scalar.scalar_factories cimport (
    make_string_scalar as cpp_make_string_scalar,
)
from pylibcudf.libcudf.strings.convert cimport (
    convert_lists as cpp_convert_lists,
)
from pylibcudf.scalar cimport Scalar
from pylibcudf.types cimport type_id

from cython.operator import dereference

__all__ = ["format_list_column"]

cpdef Column format_list_column(
    Column input,
    Scalar na_rep=None,
    Column separators=None
):
    """
    Convert a list column of strings into a formatted strings column.

    For details, see :cpp:func`cudf::strings::format_list_column`

    Parameters
    ----------
    input : Column
        Lists column to format

    na_rep : Scalar
        Replacement string for null elements.
        Default, empty string

    separators : Column
        Strings to use for enclosing list components and separating elements.
        Default, ``,``, ``[``, ``]``

    Returns
    -------
    Column
        New strings column
    """
    cdef unique_ptr[column] c_result

    if na_rep is None:
        na_rep = Scalar.from_libcudf(
            cpp_make_string_scalar("".encode())
        )

    cdef const string_scalar* c_na_rep = <const string_scalar*>(
        na_rep.c_obj.get()
    )

    if separators is None:
        separators = make_empty_column(type_id.STRING)

    with nogil:
        c_result = cpp_convert_lists.format_list_column(
            input.view(),
            dereference(c_na_rep),
            separators.view()
        )

    return Column.from_libcudf(move(c_result))
