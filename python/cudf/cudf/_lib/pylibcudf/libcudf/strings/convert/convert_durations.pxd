# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.types cimport data_type


cdef extern from "cudf/strings/convert/convert_durations.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_durations(
        const column_view & strings_col,
        data_type duration_type,
        const string & format) except +

    cdef unique_ptr[column] from_durations(
        const column_view & durations,
        const string & format) except +
