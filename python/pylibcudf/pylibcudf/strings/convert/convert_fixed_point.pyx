# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings.convert cimport (
    convert_fixed_point as cpp_fixed_point,
)
from pylibcudf.types cimport DataType, type_id

__all__ = ["from_fixed_point", "is_fixed_point", "to_fixed_point"]


cpdef Column to_fixed_point(Column input, DataType output_type):
    """
    Returns a new fixed-point column parsing decimal values from the
    provided strings column.

    For details, see :cpp:func:`cudf::strings::to_fixed_point`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    output_type : DataType
        Type of fixed-point column to return including the scale value.

    Returns
    -------
    Column
        New column of output_type.
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_fixed_point.to_fixed_point(
            input.view(),
            output_type.c_obj,
        )

    return Column.from_libcudf(move(c_result))

cpdef Column from_fixed_point(Column input):
    """
    Returns a new strings column converting the fixed-point values
    into a strings column.

    For details, see :cpp:func:`cudf::strings::from_fixed_point`

    Parameters
    ----------
    input : Column
        Fixed-point column to convert.

    Returns
    -------
    Column
        New strings column.
    """
    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_fixed_point.from_fixed_point(input.view())

    return Column.from_libcudf(move(c_result))

cpdef Column is_fixed_point(Column input, DataType decimal_type=None):
    """
    Returns a boolean column identifying strings in which all
    characters are valid for conversion to fixed-point.

    For details, see :cpp:func:`cudf::strings::is_fixed_point`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    decimal_type : DataType
        Fixed-point type (with scale) used only for checking overflow.
        Defaults to Decimal64

    Returns
    -------
    Column
        New column of boolean results for each string.
    """
    cdef unique_ptr[column] c_result

    if decimal_type is None:
        decimal_type = DataType(type_id.DECIMAL64)

    with nogil:
        c_result = cpp_fixed_point.is_fixed_point(
            input.view(),
            decimal_type.c_obj,
        )

    return Column.from_libcudf(move(c_result))
