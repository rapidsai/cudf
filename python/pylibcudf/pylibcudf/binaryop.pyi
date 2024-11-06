# Copyright (c) 2024, NVIDIA CORPORATION.

from enum import IntEnum, auto

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.types import DataType

class BinaryOperator(IntEnum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    TRUE_DIV = auto()
    FLOOR_DIV = auto()
    MOD = auto()
    PMOD = auto()
    PYMOD = auto()
    POW = auto()
    INT_POW = auto()
    LOG_BASE = auto()
    ATAN2 = auto()
    SHIFT_LEFT = auto()
    SHIFT_RIGHT = auto()
    SHIFT_RIGHT_UNSIGNED = auto()
    BITWISE_AND = auto()
    BITWISE_OR = auto()
    BITWISE_XOR = auto()
    LOGICAL_AND = auto()
    LOGICAL_OR = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS = auto()
    GREATER = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    NULL_EQUALS = auto()
    NULL_MAX = auto()
    NULL_MIN = auto()
    NULL_NOT_EQUALS = auto()
    GENERIC_BINARY = auto()
    NULL_LOGICAL_AND = auto()
    NULL_LOGICAL_OR = auto()
    INVALID_BINARY = auto()

def binary_operation(
    lhs: Column | Scalar,
    rhs: Column | Scalar,
    op: BinaryOperator,
    output_type: DataType,
) -> Column: ...
def is_supported_operation(
    out: DataType, lhs: DataType, rhs: DataType, op: BinaryOperator
) -> bool: ...
