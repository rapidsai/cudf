# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.cpp.binaryop cimport binary_operator

from .column cimport Column
from .types cimport DataType


cpdef Column binary_operation(
    object lhs,
    object rhs,
    binary_operator op,
    DataType output_type
)
