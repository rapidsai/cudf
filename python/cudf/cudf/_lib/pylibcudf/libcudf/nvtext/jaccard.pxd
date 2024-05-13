# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport size_type


cdef extern from "nvtext/jaccard.hpp" namespace "nvtext" nogil:

    cdef unique_ptr[column] jaccard_index(
        const column_view &input1,
        const column_view &input2,
        size_type width
    ) except +
