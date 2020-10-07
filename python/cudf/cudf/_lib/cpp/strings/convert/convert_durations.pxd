# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport data_type

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

cdef extern from "cudf/strings/convert/convert_durations.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] to_durations(
        const column_view & strings_col,
        data_type duration_type,
        const string & format) except +

    cdef unique_ptr[column] from_durations(
        const column_view & durations,
        const string & format) except +
