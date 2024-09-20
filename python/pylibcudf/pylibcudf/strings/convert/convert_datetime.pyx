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


cpdef Column to_timestamps(
    Column input,
    DataType timestamp_type,
    const string& format
):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_convert_datetime.to_timestamps(
            input.view(),
            timestamp_type.c_obj,
            format
        )

    return Column.from_libcudf(move(c_result))

cpdef Column from_timestamps(
    Column input,
    const string& format,
    Column input_strings_names
):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_convert_datetime.from_timestamps(
            input.view(),
            format,
            input_strings_names.view()
        )

    return Column.from_libcudf(move(c_result))

cpdef Column is_timestamp(
    Column input,
    const string& format
):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_convert_datetime.is_timestamp(
            input.view(),
            format
        )

    return Column.from_libcudf(move(c_result))
