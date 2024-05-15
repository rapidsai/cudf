# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.pylibcudf.libcudf.binaryop cimport binary_operator

from .column cimport Column
from .scalar cimport Scalar
from .types cimport DataType

# Need two separate fused types to generate the cartesian product of signatures.
ctypedef fused LeftBinaryOperand:
    Column
    Scalar

ctypedef fused RightBinaryOperand:
    Column
    Scalar


cpdef Column binary_operation(
    LeftBinaryOperand lhs,
    RightBinaryOperand rhs,
    binary_operator op,
    DataType output_type
)
