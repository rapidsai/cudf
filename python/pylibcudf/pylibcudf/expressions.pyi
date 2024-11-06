# Copyright (c) 2024, NVIDIA CORPORATION.
from enum import IntEnum, auto

from pylibcudf.scalar import Scalar

class TableReference(IntEnum):
    LEFT = auto()
    RIGHT = auto()

class ASTOperator(IntEnum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    TRUE_DIV = auto()
    FLOOR_DIV = auto()
    MOD = auto()
    PYMOD = auto()
    POW = auto()
    EQUAL = auto()
    NULL_EQUAL = auto()
    NOT_EQUAL = auto()
    LESS = auto()
    GREATER = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    BITWISE_AND = auto()
    BITWISE_OR = auto()
    BITWISE_XOR = auto()
    NULL_LOGICAL_AND = auto()
    LOGICAL_AND = auto()
    NULL_LOGICAL_OR = auto()
    LOGICAL_OR = auto()
    IDENTITY = auto()
    IS_NULL = auto()
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

class Expression: ...

class Literal(Expression):
    def __init__(self, value: Scalar) -> None: ...

class ColumnReference(Expression):
    def __init__(
        self, index: int, table_source: TableReference = TableReference.LEFT
    ) -> None: ...

class ColumnNameReference(Expression):
    def __init__(self, name: str) -> None: ...

class Operation(Expression):
    def __init__(
        self,
        op: ASTOperator,
        left: Expression,
        right: Expression | None = None,
    ) -> None: ...
