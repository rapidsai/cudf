# Copyright (c) 2024, NVIDIA CORPORATION.
from cudf._lib.cpp.types cimport size_type

from .column cimport Column


cpdef Column fill(
    object destination,
    size_type c_begin,
    size_type c_end,
    object value,
)

cpdef void fill_in_place(
    object destination,
    size_type c_begin,
    size_type c_end,
    object value,
)
