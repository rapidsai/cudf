# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings.convert cimport (
    convert_durations as cpp_convert_durations,
)

from pylibcudf.types import DataType

__all__ = ["from_durations", "to_durations"]

cpdef Column to_durations(
    Column input,
    DataType duration_type,
    str format
):
    """
    Returns a new duration column converting a strings column into
    durations using the provided format pattern.

    For details, see cpp:func:`cudf::strings::to_durations`

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    duration_type : DataType
        The duration type used for creating the output column.

    format : str
        String specifying the duration format in strings.

    Returns
    -------
    Column
        New duration column.
    """
    cdef unique_ptr[column] c_result
    cdef string c_format = format.encode()

    with nogil:
        c_result = cpp_convert_durations.to_durations(
            input.view(),
            duration_type.c_obj,
            c_format
        )

    return Column.from_libcudf(move(c_result))

cpdef Column from_durations(
    Column durations,
    str format=None
):
    """
    Returns a new strings column converting a duration column into
    strings using the provided format pattern.

    For details, see cpp:func:`cudf::strings::from_durations`

    Parameters
    ----------
    durations : Column
        Duration values to convert.

    format : str
        The string specifying output format.
        Default format is "%D days %H:%M:%S".

    Returns
    -------
    Column
        New strings column with formatted durations.
    """
    cdef unique_ptr[column] c_result

    if format is None:
        format = "%D days %H:%M:%S"
    cdef string c_format = format.encode()

    with nogil:
        c_result = cpp_convert_durations.from_durations(
            durations.view(),
            c_format
        )

    return Column.from_libcudf(move(c_result))
