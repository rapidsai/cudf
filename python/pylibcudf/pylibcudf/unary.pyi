# Copyright (c) 2024, NVIDIA CORPORATION.

from enum import IntEnum, auto

from pylibcudf.column import Column
from pylibcudf.types import DataType

class UnaryOperator(IntEnum):
    SIN = auto()
    COS = auto()
    TAN = auto()
    ARCSIN = auto()
    ARCCOS = auto()
    ARCTAN = auto()
    SINH = auto()
    COSH = auto()
    TANH = auto()
    ARCSINH = auto()
    ARCCOSH = auto()
    ARCTANH = auto()
    EXP = auto()
    LOG = auto()
    SQRT = auto()
    CBRT = auto()
    CEIL = auto()
    FLOOR = auto()
    ABS = auto()
    RINT = auto()
    BIT_INVERT = auto()
    NOT = auto()

def unary_operation(input: Column, op: UnaryOperator) -> Column: ...
def is_null(input: Column) -> Column: ...
def is_valid(input: Column) -> Column: ...
def cast(input: Column, data_type: DataType) -> Column: ...
def is_nan(input: Column) -> Column: ...
def is_not_nan(input: Column) -> Column: ...
def is_supported_cast(from_: DataType, to: DataType) -> bool: ...
