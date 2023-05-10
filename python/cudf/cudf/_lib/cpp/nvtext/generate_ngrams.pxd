# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport string_scalar
from cudf._lib.cpp.types cimport size_type


cdef extern from "nvtext/generate_ngrams.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] generate_ngrams(
        const column_view &strings,
        size_type ngrams,
        const string_scalar & separator
    ) except +

    cdef unique_ptr[column] generate_character_ngrams(
        const column_view &strings,
        size_type ngrams
    ) except +
