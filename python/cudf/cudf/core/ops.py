# Copyright (c) 2019-2020, NVIDIA CORPORATION.
from numbers import Number

import numpy as np

from cudf.core.frame import Frame

""" Global __array_ufunc__ methods
"""


def sin(arbitrary):
    if isinstance(arbitrary, Number):
        return np.sin(arbitrary)
    else:
        return getattr(arbitrary, "sin")()


def cos(arbitrary):
    if isinstance(arbitrary, Number):
        return np.cos(arbitrary)
    else:
        return getattr(arbitrary, "cos")()


def tan(arbitrary):
    if isinstance(arbitrary, Number):
        return np.tan(arbitrary)
    else:
        return getattr(arbitrary, "tan")()


def arcsin(arbitrary):
    if isinstance(arbitrary, Number):
        return np.arcsin(arbitrary)
    else:
        return getattr(arbitrary, "asin")()


def arccos(arbitrary):
    if isinstance(arbitrary, Number):
        return np.arccos(arbitrary)
    else:
        return getattr(arbitrary, "acos")()


def arctan(arbitrary):
    if isinstance(arbitrary, Number):
        return np.arctan(arbitrary)
    else:
        return getattr(arbitrary, "atan")()


def exp(arbitrary):
    if isinstance(arbitrary, Number):
        return np.exp(arbitrary)
    else:
        return getattr(arbitrary, "exp")()


def log(arbitrary):
    if isinstance(arbitrary, Number):
        return np.log(arbitrary)
    else:
        return getattr(arbitrary, "log")()


def sqrt(arbitrary):
    if isinstance(arbitrary, Number):
        return np.sqrt(arbitrary)
    else:
        return getattr(arbitrary, "sqrt")()


def logical_not(arbitrary):
    if isinstance(arbitrary, Number):
        return np.logical_not(arbitrary)
    else:
        return getattr(arbitrary, "logical_not")()


def logical_and(lhs, rhs):
    if isinstance(lhs, Number) and isinstance(rhs, Number):
        return np.logical_and(lhs, rhs)
    else:
        return getattr(lhs, "logical_and")(rhs)


def logical_or(lhs, rhs):
    if isinstance(lhs, Number) and isinstance(rhs, Number):
        return np.logical_or(lhs, rhs)
    else:
        return getattr(lhs, "logical_or")(rhs)


def remainder(lhs, rhs):
    if isinstance(lhs, Number) and isinstance(rhs, Number):
        return np.mod(lhs, rhs)
    elif isinstance(lhs, Frame):
        return getattr(lhs, "remainder")(rhs)
    else:
        return getattr(rhs, "__rmod__")(lhs)


def floor_divide(lhs, rhs):
    if isinstance(lhs, Number) and isinstance(rhs, Number):
        return np.floor_divide(lhs, rhs)
    elif isinstance(lhs, Frame):
        return getattr(lhs, "floordiv")(rhs)
    else:
        return getattr(rhs, "__rfloordiv__")(lhs)


def subtract(lhs, rhs):
    if isinstance(lhs, Number) and isinstance(rhs, Number):
        return np.subtract(lhs, rhs)
    elif isinstance(lhs, Frame):
        return getattr(lhs, "__sub__")(rhs)
    else:
        return getattr(rhs, "__rsub__")(lhs)


def add(lhs, rhs):
    if isinstance(lhs, Number) and isinstance(rhs, Number):
        return np.add(lhs, rhs)
    elif isinstance(rhs, Frame):
        return getattr(rhs, "__radd__")(lhs)
    else:
        return getattr(lhs, "__add__")(rhs)


def true_divide(lhs, rhs):
    if isinstance(lhs, Number) and isinstance(rhs, Number):
        return np.true_divide(lhs, rhs)
    elif isinstance(rhs, Frame):
        return getattr(rhs, "__rtruediv__")(lhs)
    else:
        return getattr(lhs, "__truediv__")(rhs)


def multiply(lhs, rhs):
    if isinstance(lhs, Number) and isinstance(rhs, Number):
        return np.multiply(lhs, rhs)
    elif isinstance(rhs, Frame):
        return getattr(rhs, "__rmul__")(lhs)
    else:
        return getattr(lhs, "__mul__")(rhs)
