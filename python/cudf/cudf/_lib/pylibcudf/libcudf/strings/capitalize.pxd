# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.strings.char_types cimport (
    string_character_types,
)


cdef extern from "cudf/strings/capitalize.hpp" namespace "cudf::strings" nogil:
    cdef unique_ptr[column] capitalize(
        const column_view & strings,
        const string_scalar & delimiters
        ) except +

    cdef unique_ptr[column] title(
        const column_view & strings,
        string_character_types sequence_type
        ) except +

    cdef unique_ptr[column] is_title(
        const column_view & strings) except +
