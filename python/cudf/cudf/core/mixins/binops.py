# Copyright (c) 2022, NVIDIA CORPORATION.

from .mixin_factory import _create_delegating_mixin

BinaryOperand = _create_delegating_mixin(
    "BinaryOperand",
    "Mixin encapsulating binary operations.",
    "BINARY_OPERATION",
    "_binaryop",
    {
        # Numeric operations.
        "__add__",
        "__sub__",
        "__mul__",
        "__matmul__",
        "__truediv__",
        "__floordiv__",
        "__mod__",
        # "__divmod__", # Not yet implemented
        "__pow__",
        # "__lshift__", # Not yet implemented
        # "__rshift__", # Not yet implemented
        "__and__",
        "__xor__",
        "__or__",
        # Reflected numeric operations.
        "__radd__",
        "__rsub__",
        "__rmul__",
        "__rmatmul__",
        "__rtruediv__",
        "__rfloordiv__",
        "__rmod__",
        # "__rdivmod__", # Not yet implemented
        "__rpow__",
        # "__rlshift__", # Not yet implemented
        # "__rrshift__", # Not yet implemented
        "__rand__",
        "__rxor__",
        "__ror__",
        # Rich comparison operations.
        "__lt__",
        "__le__",
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
    },
)


def _is_reflected_op(op):
    return op[2] == "r" and op != "__rshift__"


BinaryOperand._is_reflected_op = staticmethod(_is_reflected_op)
