# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from pylibcudf.column cimport Column
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.strings.convert cimport (
    convert_datetime as cpp_convert_datetime,
)
from pylibcudf.utils cimport _get_stream, _get_memory_resource
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.types import DataType

__all__ = ["from_timestamps", "is_timestamp", "to_timestamps"]

cpdef Column to_timestamps(
    Column input,
    DataType timestamp_type,
    str format,
    Stream stream=None,
    DeviceMemoryResource mr=None,
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

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New datetime column
    """
    cdef unique_ptr[column] c_result
    cdef string c_format = format.encode()
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)
    with nogil:
        c_result = cpp_convert_datetime.to_timestamps(
            input.view(),
            timestamp_type.c_obj,
            c_format,
            stream.view(),
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), stream, mr)

cpdef Column from_timestamps(
    Column timestamps,
    str format,
    Column input_strings_names,
    Stream stream=None,
    DeviceMemoryResource mr=None,
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

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New strings column with formatted timestamps.
    """
    cdef unique_ptr[column] c_result
    cdef string c_format = format.encode()
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)
    with nogil:
        c_result = cpp_convert_datetime.from_timestamps(
            timestamps.view(),
            c_format,
            input_strings_names.view(),
            stream.view(),
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), stream, mr)

cpdef Column is_timestamp(
    Column input,
    str format,
    Stream stream=None,
    DeviceMemoryResource mr=None,
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

    stream : Stream | None
        CUDA stream on which to perform the operation.

    Returns
    -------
    Column
        New bool column.
    """
    cdef unique_ptr[column] c_result
    cdef string c_format = format.encode()
    stream = _get_stream(stream)
    mr = _get_memory_resource(mr)
    with nogil:
        c_result = cpp_convert_datetime.is_timestamp(
            input.view(),
            c_format,
            stream.view(),
            mr.get_mr()
        )

    return Column.from_libcudf(move(c_result), stream, mr)
