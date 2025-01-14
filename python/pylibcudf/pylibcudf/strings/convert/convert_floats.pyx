# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings.convert cimport (
    convert_floats as cpp_convert_floats,
)
from pylibcudf.types cimport DataType

__all__ = ["from_floats", "is_float", "to_floats"]

cpdef Column to_floats(Column strings, DataType output_type):
    """
    Returns a new numeric column by parsing float values from each string
    in the provided strings column.

    For details, see cpp:func:`cudf::strings::to_floats`

    Parameters
    ----------
    strings : Column
        Strings instance for this operation.

    output_type : DataType
        Type of float numeric column to return.

    Returns
    -------
    Column
        New column with floats converted from strings.
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_convert_floats.to_floats(
            strings.view(),
            output_type.c_obj,
        )

    return Column.from_libcudf(move(c_result))


cpdef Column from_floats(Column floats):
    """
    Returns a new strings column converting the float values from the
    provided column into strings.

    For details, see cpp:func:`cudf::strings::from_floats`

    Parameters
    ----------
    floats : Column
        Numeric column to convert.

    Returns
    -------
    Column
        New strings column with floats as strings.
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_convert_floats.from_floats(floats.view())

    return Column.from_libcudf(move(c_result))


cpdef Column is_float(Column input):
    """
    Returns a boolean column identifying strings in which all
    characters are valid for conversion to floats.

    For details, see cpp:func:`cudf::strings::is_float`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    Returns
    -------
    Column
        New column of boolean results for each string.
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_convert_floats.is_float(input.view())

    return Column.from_libcudf(move(c_result))
