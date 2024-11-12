# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings.convert cimport (
    convert_datetime as cpp_convert_datetime,
)

from pylibcudf.types import DataType

__all__ = ["from_timestamps", "is_timestamp", "to_timestamps"]

cpdef Column to_timestamps(
    Column input,
    DataType timestamp_type,
    str format
):
    """
    Returns a new timestamp column converting a strings column into
    timestamps using the provided format pattern.

    For details, see cpp:`cudf::strings::to_timestamps`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    timestamp_type : DataType
        The timestamp type used for creating the output column.

    format : str
        String specifying the timestamp format in strings.

    Returns
    -------
    Column
        New datetime column
    """
    cdef unique_ptr[column] c_result
    cdef string c_format = format.encode()
    with nogil:
        c_result = cpp_convert_datetime.to_timestamps(
            input.view(),
            timestamp_type.c_obj,
            c_format
        )

    return Column.from_libcudf(move(c_result))

cpdef Column from_timestamps(
    Column timestamps,
    str format,
    Column input_strings_names
):
    """
    Returns a new strings column converting a timestamp column into
    strings using the provided format pattern.

    For details, see cpp:`cudf::strings::from_timestamps`.

    Parameters
    ----------
    timestamps : Column
        Timestamp values to convert

    format : str
        The string specifying output format.

    input_strings_names : Column
        The string names to use for weekdays ("%a", "%A") and months ("%b", "%B").

    Returns
    -------
    Column
        New strings column with formatted timestamps.
    """
    cdef unique_ptr[column] c_result
    cdef string c_format = format.encode()
    with nogil:
        c_result = cpp_convert_datetime.from_timestamps(
            timestamps.view(),
            c_format,
            input_strings_names.view()
        )

    return Column.from_libcudf(move(c_result))

cpdef Column is_timestamp(
    Column input,
    str format
):
    """
    Verifies the given strings column can be parsed to timestamps
    using the provided format pattern.

    For details, see cpp:`cudf::strings::is_timestamp`.

    Parameters
    ----------
    input : Column
        Strings instance for this operation.

    format : str
        String specifying the timestamp format in strings.

    Returns
    -------
    Column
        New bool column.
    """
    cdef unique_ptr[column] c_result
    cdef string c_format = format.encode()
    with nogil:
        c_result = cpp_convert_datetime.is_timestamp(
            input.view(),
            c_format
        )

    return Column.from_libcudf(move(c_result))
