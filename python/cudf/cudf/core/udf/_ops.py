import math
import operator

from cudf.core.dtypes import CategoricalDtype
from cudf.utils.dtypes import (
    BOOL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
    TIMEDELTA_TYPES,
)

arith_ops = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow,
]

unary_ops = [
    math.acos,
    math.acosh,
    math.asin,
    math.asinh,
    math.atan,
    math.atanh,
    math.ceil,
    math.cos,
    math.degrees,
    math.erf,
    math.erfc,
    math.exp,
    math.expm1,
    math.fabs,
    math.floor,
    math.gamma,
    math.lgamma,
    math.log,
    math.log10,
    math.log1p,
    math.log2,
    math.radians,
    math.sin,
    math.sinh,
    math.sqrt,
    math.tan,
    math.tanh,
    operator.pos,
    operator.neg,
    operator.not_,
    operator.invert,
]

comparison_ops = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
]


# currently only numeric types are supported.
SUPPORTED_TYPES = NUMERIC_TYPES | BOOL_TYPES | DATETIME_TYPES | TIMEDELTA_TYPES


def _is_jit_supported_type(dtype):
    # category dtype isn't hashable
    if isinstance(dtype, CategoricalDtype):
        return False
    return str(dtype) in SUPPORTED_TYPES
