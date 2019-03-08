
import numpy as np
from numbers import Number


def sqrt(arbitrary):
    if isinstance(arbitrary, Number):
        return np.sqrt(arbitrary)
    else:
        return arbitrary._unaryop('sqrt')
