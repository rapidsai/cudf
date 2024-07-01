# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.strings.side_type cimport side_type


cdef extern from "cudf/strings/strip.hpp" namespace "cudf::strings" nogil:

    cdef unique_ptr[column] strip(
        column_view source_strings,
        side_type stype,
        string_scalar to_strip) except +
