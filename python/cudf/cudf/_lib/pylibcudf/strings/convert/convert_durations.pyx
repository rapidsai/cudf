# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.strings.convert cimport (
    convert_durations as cpp_convert_durations,
)

from cudf._lib.pylibcudf.types import DataType


cpdef Column to_durations(
    Column input,
    DataType duration_type,
    const string& format
):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_convert_durations.to_durations(
            input.view(),
            duration_type.c_obj,
            format
        )

    return Column.from_libcudf(move(c_result))

cpdef Column from_durations(
    Column input,
    const string& format
):
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = cpp_convert_durations.from_durations(
            input.view(),
            format
        )

    return Column.from_libcudf(move(c_result))
