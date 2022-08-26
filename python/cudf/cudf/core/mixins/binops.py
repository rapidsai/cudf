# Copyright (c) 2022, NVIDIA CORPORATION.

from .mixin_factory import Operation, _create_delegating_mixin

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

# TODO: See if there is a better approach to these two issues: 1) The mixin
# assumes a single standard parameter, whereas binops have two, and 2) we need
# a way to determine reflected vs normal ops.


def _binaryop(self, other, op: str):
    """The core binary_operation function.

    Must be overridden by subclasses, the default implementation raises a
    NotImplementedError.
    """
    if op == "__eq__":
        raise TypeError(
            "'==' not supported between instances of "
            f"'{type(self).__name__}' and '{type(other).__name__}'"
        )
    if op == "__ne__":
        raise TypeError(
            "'!=' not supported between instances of "
            f"'{type(self).__name__}' and '{type(other).__name__}'"
        )
    raise NotImplementedError()


def _check_reflected_op(op):
    if reflect := op[2] == "r" and op != "__rshift__":
        op = op[:2] + op[3:]
    return reflect, op


BinaryOperand._binaryop = _binaryop
BinaryOperand._check_reflected_op = staticmethod(_check_reflected_op)

# It is necessary to override the default object.__eq__ so that objects don't
# automatically support equality binops using the wrong operator implementation
# (falling back to ``object``). We must override the object.__eq__ with an
# Operation rather than a plain function/method, so that the BinaryOperand
# mixin overrides it for classes that define their own __eq__ in _binaryop.
BinaryOperand.__eq__ = Operation("__eq__", {}, BinaryOperand._binaryop)
BinaryOperand.__ne__ = Operation("__ne__", {}, BinaryOperand._binaryop)
