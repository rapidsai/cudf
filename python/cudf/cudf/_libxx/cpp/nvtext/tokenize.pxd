# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.scalar.scalar cimport string_scalar

cdef extern from "nvtext/tokenize.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] tokenize(
        const column_view & strings,
        const string_scalar & delimiter
    ) except +

    cdef unique_ptr[column] tokenize_multi "nvtext::tokenize" (
        const column_view & strings,
        const column_view & delimiters
    ) except +

    cdef unique_ptr[column] count_tokens(
        const column_view & strings,
        const string_scalar & delimiter
    ) except +

    cdef unique_ptr[column] count_tokens_multi "nvtext::count_tokens" (
        const column_view & strings,
        const column_view & delimiters
    ) except +
