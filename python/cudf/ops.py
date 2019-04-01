# Copyright (c) 2019, NVIDIA CORPORATION.

import numpy as np
from numbers import Number

""" Global __array_ufunc__ methods
"""


def sin(arbitrary):
    if isinstance(arbitrary, Number):
        return np.sin(arbitrary)
    else:
        return arbitrary._unaryop('sin')


def cos(arbitrary):
    if isinstance(arbitrary, Number):
        return np.cos(arbitrary)
    else:
        return arbitrary._unaryop('cos')


def tan(arbitrary):
    if isinstance(arbitrary, Number):
        return np.tan(arbitrary)
    else:
        return arbitrary._unaryop('tan')


def arcsin(arbitrary):
    if isinstance(arbitrary, Number):
        return np.arcsin(arbitrary)
    else:
        return arbitrary._unaryop('asin')


def arccos(arbitrary):
    if isinstance(arbitrary, Number):
        return np.arccos(arbitrary)
    else:
        return arbitrary._unaryop('acos')


def arctan(arbitrary):
    if isinstance(arbitrary, Number):
        return np.arctan(arbitrary)
    else:
        return arbitrary._unaryop('atan')


def exp(arbitrary):
    if isinstance(arbitrary, Number):
        return np.exp(arbitrary)
    else:
        return arbitrary._unaryop('exp')


def log(arbitrary):
    if isinstance(arbitrary, Number):
        return np.log(arbitrary)
    else:
        return arbitrary._unaryop('log')


def sqrt(arbitrary):
    if isinstance(arbitrary, Number):
        return np.sqrt(arbitrary)
    else:
        return arbitrary._unaryop('sqrt')
