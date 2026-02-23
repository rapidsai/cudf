# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
        # "__divmod__", # Implemented on BinaryOperand directly
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
        # "__rdivmod__", # Implemented on BinaryOperand directly
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

# TODO: See if there is a better approach to these two issues: 1) The mixin
# assumes a single standard parameter, whereas binops have two, and 2) we need
# a way to determine reflected vs normal ops.


def _binaryop(self, other, op: str):
    """The core binary_operation function.

    Must be overridden by subclasses, the default implementation raises a
    NotImplementedError.
    """
    raise NotImplementedError


def _check_reflected_op(op):
    if reflect := op[2] == "r" and op != "__rshift__":
        op = op[:2] + op[3:]
    return reflect, op


BinaryOperand._binaryop = _binaryop
BinaryOperand._check_reflected_op = staticmethod(_check_reflected_op)


def _divmod(self, other):
    return self.__floordiv__(other), self.__mod__(other)


def _rdivmod(self, other):
    return self.__rfloordiv__(other), self.__rmod__(other)


_binaryoperand_init_subclass = BinaryOperand.__init_subclass__


@classmethod
def _binaryoperand_init_subclass_with_divmod(cls) -> None:
    _binaryoperand_init_subclass.__func__(cls)

    valid_operations: set[str] = set()
    for base_cls in cls.__mro__:
        valid_operations |= getattr(
            base_cls, "_VALID_BINARY_OPERATIONS", set()
        )

    if (
        "__floordiv__" in valid_operations
        and "__mod__" in valid_operations
        and "__divmod__" not in cls.__dict__
    ):
        cls.__divmod__ = _divmod

    if (
        "__rfloordiv__" in valid_operations
        and "__rmod__" in valid_operations
        and "__rdivmod__" not in cls.__dict__
    ):
        cls.__rdivmod__ = _rdivmod


BinaryOperand.__init_subclass__ = _binaryoperand_init_subclass_with_divmod
