# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view

cdef extern from "nvtext/stemmer.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] porter_stemmer_measure(
        const column_view & strings
    ) except +
