# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.types cimport size_type


cdef extern from "nvtext/replace.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] replace_tokens(
        const column_view & strings,
        const column_view & targets,
        const column_view & replacements,
        const string_scalar & delimiter
    ) except +

    cdef unique_ptr[column] filter_tokens(
        const column_view & strings,
        size_type min_token_length,
        const string_scalar & replacement,
        const string_scalar & delimiter
    ) except +
