# Copyright (c) 2019, NVIDIA CORPORATION.
from numbers import Number

import numpy as np

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
    else:
        return getattr(lhs, "remainder")(rhs)


def floor_divide(lhs, rhs):
    if isinstance(lhs, Number) and isinstance(rhs, Number):
        return np.floor_divide(lhs, rhs)
    else:
        return getattr(lhs, "floordiv")(rhs)
