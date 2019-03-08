# Copyright (c) 2019, NVIDIA CORPORATION.

import numpy as np
from numbers import Number

""" Global __array_ufunc__ methods
"""


def sqrt(arbitrary):
    if isinstance(arbitrary, Number):
        return np.sqrt(arbitrary)
    else:
        return arbitrary._unaryop('sqrt')
