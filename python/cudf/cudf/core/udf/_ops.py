import math
import operator

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
    # Trigonometry
    math.sin,
    math.cos,
    math.tan,
    math.asin,
    math.acos,
    math.atan,
    # Rounding
    math.ceil,
    math.floor,
    # Arithmetic
    math.sqrt,
    # Sign
    operator.pos,
    operator.neg,
    # Bit
    operator.not_,
]

comparison_ops = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
]
