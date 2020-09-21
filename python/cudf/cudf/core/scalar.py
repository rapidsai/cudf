# Copyright (c) 2020, NVIDIA CORPORATION.
import cudf._lib as libcudf
from cudf.utils.dtypes import to_cudf_compatible_scalar
from numpy import find_common_type
import numpy as np

class Scalar(libcudf.scalar.Scalar):
    def __init__(self, value, dtype=None):

        # When a scalar is instantiated from python, 
        # cache on the host until needed in libcudf
        # Not synced, so set _device_sync = False

        # when the value is requested in libcudf, 
        # perform host-device transfer
        # set device-host sync flag to True

        # When a scalar is received from a libcudf function
        # value is not cached on the host
        # _device_sync = False in that case
        
        # When the scalar is needed in python
        # perform a device-to-host transfer
        # set _device-host sync flag to True

        #self._host_value = value
        #self._device_value_is_current = False
        # Do not immediately create 



        if isinstance(value, libcudf.scalar.Scalar):
            if dtype and not value.dtype == dtype:
                # TODO should be doable on the device
                value = libcudf.scalar.Scalar(value.value, dtype=dtype)
            self._data = value
        else:
            self._data = libcudf.scalar.Scalar(value, dtype=dtype)

    @property
    def value(self):
        return self._data.value

    def __index__(self):
        if not self.dtype.kind in {'u', 'i'}:
            raise TypeError("Only Integer typed scalars may be used in slices")
        return int(self)

    @property
    def ptr(self):
        return self._data.ptr

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def is_valid(self):
        return self._data.is_valid

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __bool__(self):
        return bool(self.value)

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
        return self._scalar_binop(other, "__gt__").value

    def __lt__(self, other):
        return self._scalar_binop(other, "__lt__").value

    def __ge__(self, other):
        return self._scalar_binop(other, "__ge__").value

    def __le__(self, other):
        return self._scalar_binop(other, "__le__").value

    def __eq__(self, other):
        return self._scalar_binop(other, '__eq__').value

    def __ne__(self, other):
        return self._scalar_binop(other, "__ne__").value

    def __abs__(self):
        return self._scalar_unaop('__abs__')

    def __round__(self, n):
        return self._scalar_binop(n, '__round__')
    
    __div__ = __truediv__

    def _binop_result_dtype_or_error(self, other, op):

        if (self.dtype.kind == "O" and other.dtype.kind != "O") or (
            self.dtype.kind != "O" and other.dtype.kind == "O"
        ):
            wrong_dtype = self.dtype if self.dtype.kind != "O" else other.dtype
            raise TypeError(
                f"Can only concatenate string (not {wrong_dtype}) to string"
            )
        if (self.dtype.kind == "O" or other.dtype.kind == "O") and op != "__add__":
            raise TypeError(f"{op} is not supported for string type scalars")

        return find_common_type([self.dtype, other.dtype], [])

    def _scalar_binop(self, other, op):
        other = to_cudf_compatible_scalar(other)

        if op in ["__eq__", "__ne__", "__lt__", "__gt__", "__le__", "__ge__"]:
            out_dtype = np.bool
        else:
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

    def _scalar_unaop(self, op):
        return Scalar(getattr(self.value, op)())
