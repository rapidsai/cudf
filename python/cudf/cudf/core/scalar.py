# Copyright (c) 2020, NVIDIA CORPORATION.
import cudf._lib as libcudf
from cudf.utils.dtypes import to_cudf_compatible_scalar, INVALID_BINOP_DTYPE_COMBOS
from numpy import find_common_type
import numpy as np
from cudf.core.series import truediv_int_dtype_corrections

class Scalar(libcudf.scalar.Scalar):
    def __init__(self, value, dtype=None):
        super().__init__(value, dtype=dtype)

    def __index__(self):
        if not self.dtype.kind in {'u', 'i'}:
            raise TypeError("Only Integer typed scalars may be used in slices")
        return int(self)

    def __int__(self):
        return int(self.value)
    
    def __float__(self):
        return float(self.value)

    def __bool__(self):
        return bool(self.value)

    # Scalar Binary Operations
    def __add__(self, other):
        return self._scalar_binop(other, "__add__")

    def __radd__(self, other):
        return self._scalar_binop(other, '__radd__')

    def __sub__(self, other):
        return self._scalar_binop(other, "__sub__")

    def __rsub__(self, other):
        return self._scalar_binop(other, "__rsub__")

    def __mul__(self, other):
        return self._scalar_binop(other, "__mul__")

    def __rmul__(self, other):
        return self._scalar_binop(other, "__rmul__")

    def __truediv__(self, other):
        return self._scalar_binop(other, "__truediv__")
    
    def __floordiv__(self, other):
        return self._scalar_binop(other, '__floordiv__')

    def __rtruediv__(self, other):
        return self._scalar_binop(other, "__rtruediv__")

    def __mod__(self, other):
        return self._scalar_binop(other, "__mod__")

    def __divmod__(self, other):
        return self._scalar_binop(other, "__divmod__")

    def __and__(self, other):
        return self._scalar_binop(other, "__and__")

    def __xor__(self, other):
        return self._scalar_binop(other, "__or__")

    def __pow__(self, other):
        return self._scalar_binop(other, "__pow__")

    def __gt__(self, other):
        return self._scalar_binop(other, "__gt__")

    def __lt__(self, other):
        return self._scalar_binop(other, "__lt__")

    def __ge__(self, other):
        return self._scalar_binop(other, "__ge__")

    def __le__(self, other):
        return self._scalar_binop(other, "__le__")

    def __eq__(self, other):
        return self._scalar_binop(other, '__eq__')

    def __ne__(self, other):
        return self._scalar_binop(other, "__ne__")

    def __round__(self, n):
        return self._scalar_binop(n, '__round__')

    # Scalar Unary Operations
    def __abs__(self):
        return self._scalar_unaop('__abs__')

    def __ceil__(self):
        return self._scalar_unaop('__ceil__')

    def __floor__(self):
        return self._scalar_unaop('__floor__')

    def __invert__(self):
        return self._scalar_unaop('__invert__')

    def __neg__(self):
        return self._scalar_unaop('__neg__')

    def _binop_result_dtype_or_error(self, other, op):
        if op in ["__eq__", "__ne__", "__lt__", "__gt__", "__le__", "__ge__"]:
            return np.bool

        if (self.dtype.name, other.dtype.name) in INVALID_BINOP_DTYPE_COMBOS[op]:
            raise TypeError(f"{op} not supported between "\
                            f"{self.dtype} and {other.dtype} scalars")

        if op == '__truediv__':
            # todo - this doesn't work out quite right w.r.t numpy 
            dtype_l = truediv_int_dtype_corrections.get(self.dtype.name, self.dtype.name)
            dtype_r = truediv_int_dtype_corrections.get(other.dtype.name, other.dtype.name)

            if dtype_l == 'float32' and dtype_r == 'float32':
                return 'float32'
            else:
                return 'float64'

        return find_common_type([self.dtype, other.dtype], [])

    def _scalar_binop(self, other, op):
        other = to_cudf_compatible_scalar(other)
        out_dtype = self._binop_result_dtype_or_error(other, op)
        valid = self.is_valid() and (
            isinstance(other, np.generic) or other.is_valid()
        )
        if not valid:
            return Scalar(None, dtype=out_dtype)
        else:
            result = self._dispatch_scalar_binop(other, op)
            return Scalar(result, dtype=out_dtype)

    def _dispatch_scalar_binop(self, other, op):
        if isinstance(other, Scalar):
            other = other.value
        return getattr(self.value, op)(other)

    def _unaop_result_type_or_error(self, op):
        if op == '__neg__' and self.dtype == 'bool':
            raise TypeError("Boolean scalars in cuDF, do not support" \
                            "negation, use logical not")
        if op in {'__ceil__', '__floor__'} and self.dtype == 'int8':
            return np.dtype('float32')
        else:
            return self.dtype

    def _scalar_unaop(self, op):
        out_dtype = self._unaop_result_type_or_error(op)
        if not self.is_valid():
            result = None
        else:
            result = self._dispatch_scalar_unaop(op)
            return Scalar(result, dtype=out_dtype)

    def _dispatch_scalar_unaop(self, op):
        if op == '__floor__':
            return np.floor(self.value)
        if op == '__ceil__':
            return np.ceil(self.value)
        return getattr(self.value, op)()

class cudf_NA_type(object):
    def __init__(self):
        pass
    def __repr__(self):
        return "<NA>"
    def __bool__(self):
        raise TypeError("boolean value of cudf.NA is ambiguous")

NA = cudf_NA_type()
