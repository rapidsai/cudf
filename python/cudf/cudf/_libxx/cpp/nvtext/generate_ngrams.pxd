# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.scalar.scalar cimport scalar
from cudf._libxx.cpp.types cimport size_type

cdef extern from "nvtext/generate_ngrams.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] generate_ngrams(
        const column_view &strings,
        size_type ngrams,
        const scalar & separator
    ) except +
