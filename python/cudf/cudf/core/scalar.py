# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import decimal
import operator
from collections import OrderedDict

import numpy as np
import pyarrow as pa

import cudf
from cudf.api.types import is_scalar
from cudf.core.dtypes import ListDtype, StructDtype
from cudf.core.missing import NA, NaT
from cudf.core.mixins import BinaryOperand
from cudf.utils.dtypes import (
    get_allowed_combinations_for_operator,
    to_cudf_compatible_scalar,
)


# Note that the metaclass below can easily be generalized for use with
# other classes, if needed in the future. Simply replace the arguments
# of the `__call__` method with `*args` and `**kwargs`. This will
# result in additional overhead when constructing the cache key, as
# unpacking *args and **kwargs is not cheap. See the discussion in
# https://github.com/rapidsai/cudf/pull/11246#discussion_r955843532
# for details.
class CachedScalarInstanceMeta(type):
    """
    Metaclass for Scalar that caches `maxsize` instances.

    After `maxsize` is reached, evicts the least recently used
    instances to make room for new values.
    """

    def __new__(cls, names, bases, attrs, **kwargs):
        return type.__new__(cls, names, bases, attrs)

    # choose 128 because that's the default `maxsize` for
    # `functools.lru_cache`:
    def __init__(self, names, bases, attrs, maxsize=128):
        self.__maxsize = maxsize
        self.__instances = OrderedDict()

    def __call__(self, value, dtype=None):
        # the cache key is constructed from the arguments, and also
        # the _types_ of the arguments, since objects of different
        # types can compare equal
        cache_key = (value, type(value), dtype, type(dtype))
        try:
            # try retrieving an instance from the cache:
            self.__instances.move_to_end(cache_key)
            return self.__instances[cache_key]
        except KeyError:
            # if an instance couldn't be found in the cache,
            # construct it and add to cache:
            obj = super().__call__(value, dtype=dtype)
            try:
                self.__instances[cache_key] = obj
            except TypeError:
                # couldn't hash the arguments, don't cache:
                return obj
            if len(self.__instances) > self.__maxsize:
                self.__instances.popitem(last=False)
            return obj
        except TypeError:
            # couldn't hash the arguments, don't cache:
            return super().__call__(value, dtype=dtype)

    def _clear_instance_cache(self):
        self.__instances.clear()


class Scalar(BinaryOperand, metaclass=CachedScalarInstanceMeta):
    """
    A GPU-backed scalar object with NumPy scalar like properties
    May be used in binary operations against other scalars, cuDF
    Series, DataFrame, and Index objects.

    Examples
    --------
    >>> import cudf
    >>> cudf.Scalar(42, dtype='int64')
    Scalar(42, dtype=int64)
    >>> cudf.Scalar(42, dtype='int32') + cudf.Scalar(42, dtype='float64')
    Scalar(84.0, dtype=float64)
    >>> cudf.Scalar(42, dtype='int64') + np.int8(21)
    Scalar(63, dtype=int64)
    >>> x = cudf.Scalar(42, dtype='datetime64[s]')
    >>> y = cudf.Scalar(21, dtype='timedelta64[ns]')
    >>> x - y
    Scalar(1970-01-01T00:00:41.999999979, dtype=datetime64[ns])
    >>> cudf.Series([1,2,3]) + cudf.Scalar(1)
    0    2
    1    3
    2    4
    dtype: int64
    >>> df = cudf.DataFrame({'a':[1,2,3], 'b':[4.5, 5.5, 6.5]})
    >>> slr = cudf.Scalar(10, dtype='uint8')
    >>> df - slr
       a    b
    0 -9 -5.5
    1 -8 -4.5
    2 -7 -3.5

    Parameters
    ----------
    value : Python Scalar, NumPy Scalar, or cuDF Scalar
        The scalar value to be converted to a GPU backed scalar object
    dtype : np.dtype or string specifier
        The data type
    """

    _VALID_BINARY_OPERATIONS = BinaryOperand._SUPPORTED_BINARY_OPERATIONS

    def __init__(self, value, dtype=None):
        self._host_value = None
        self._host_dtype = None
        self._device_value = None

        if isinstance(value, Scalar):
            if value._is_host_value_current:
                self._host_value = value._host_value
                self._host_dtype = value._host_dtype
            else:
                self._device_value = value._device_value
        else:
            self._host_value, self._host_dtype = self._preprocess_host_value(
                value, dtype
            )

    @classmethod
    def from_device_scalar(cls, device_scalar):
        if not isinstance(device_scalar, cudf._lib.scalar.DeviceScalar):
            raise TypeError(
                "Expected an instance of DeviceScalar, "
                f"got {type(device_scalar).__name__}"
            )
        obj = object.__new__(cls)
        obj._host_value = None
        obj._host_dtype = None
        obj._device_value = device_scalar
        return obj

    @property
    def _is_host_value_current(self):
        return self._host_value is not None

    @property
    def _is_device_value_current(self):
        return self._device_value is not None

    @property
    def device_value(self):
        if self._device_value is None:
            self._device_value = cudf._lib.scalar.DeviceScalar(
                self._host_value, self._host_dtype
            )
        return self._device_value

    @property
    def value(self):
        if not self._is_host_value_current:
            self._device_value_to_host()
        return self._host_value

    # todo: change to cached property
    @property
    def dtype(self):
        if self._is_host_value_current:
            if isinstance(self._host_value, str):
                return cudf.dtype("object")
            else:
                return self._host_dtype
        else:
            return self.device_value.dtype

    def is_valid(self):
        if not self._is_host_value_current:
            self._device_value_to_host()
        return not cudf._lib.scalar._is_null_host_scalar(self._host_value)

    def _device_value_to_host(self):
        self._host_value = self._device_value._to_host_scalar()

    def _preprocess_host_value(self, value, dtype):
        valid = not cudf._lib.scalar._is_null_host_scalar(value)

        if isinstance(value, list):
            if dtype is not None:
                raise TypeError("Lists may not be cast to a different dtype")
            else:
                dtype = ListDtype.from_arrow(
                    pa.infer_type([value], from_pandas=True)
                )
                return value, dtype
        elif isinstance(dtype, ListDtype):
            if value not in {None, NA}:
                raise ValueError(f"Can not coerce {value} to ListDtype")
            else:
                return NA, dtype

        if isinstance(value, dict):
            if dtype is None:
                dtype = StructDtype.from_arrow(
                    pa.infer_type([value], from_pandas=True)
                )
            return value, dtype
        elif isinstance(dtype, StructDtype):
            if value not in {None, NA}:
                raise ValueError(f"Can not coerce {value} to StructDType")
            else:
                return NA, dtype

        if isinstance(dtype, cudf.core.dtypes.DecimalDtype):
            value = pa.scalar(
                value, type=pa.decimal128(dtype.precision, dtype.scale)
            ).as_py()
        if isinstance(value, decimal.Decimal) and dtype is None:
            dtype = cudf.Decimal128Dtype._from_decimal(value)

        value = to_cudf_compatible_scalar(value, dtype=dtype)

        if dtype is None:
            if not valid:
                if value is NaT:
                    value = value.to_numpy()

                if isinstance(value, (np.datetime64, np.timedelta64)):
                    unit, _ = np.datetime_data(value)
                    if unit == "generic":
                        raise TypeError(
                            "Cant convert generic NaT to null scalar"
                        )
                    else:
                        dtype = value.dtype
                else:
                    raise TypeError(
                        "dtype required when constructing a null scalar"
                    )
            else:
                dtype = value.dtype

        if not isinstance(dtype, cudf.core.dtypes.DecimalDtype):
            dtype = cudf.dtype(dtype)

        if not valid:
            value = NaT if dtype.kind in "mM" else NA

        return value, dtype

    def _sync(self):
        """
        If the cache is not synched, copy either the device or host value
        to the host or device respectively. If cache is valid, do nothing
        """
        if self._is_host_value_current and self._is_device_value_current:
            return
        elif self._is_host_value_current and not self._is_device_value_current:
            self._device_value = cudf._lib.scalar.DeviceScalar(
                self._host_value, self._host_dtype
            )
        elif self._is_device_value_current and not self._is_host_value_current:
            self._host_value = self._device_value.value
            self._host_dtype = self._host_value.dtype
        else:
            raise ValueError("Invalid cudf.Scalar")

    def __index__(self):
        if self.dtype.kind not in {"u", "i"}:
            raise TypeError("Only Integer typed scalars may be used in slices")
        return int(self)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __bool__(self):
        return bool(self.value)

    def __round__(self, n):
        return self._binaryop(n, "__round__")

    # Scalar Unary Operations
    def __abs__(self):
        return self._scalar_unaop("__abs__")

    def __ceil__(self):
        return self._scalar_unaop("__ceil__")

    def __floor__(self):
        return self._scalar_unaop("__floor__")

    def __invert__(self):
        return self._scalar_unaop("__invert__")

    def __neg__(self):
        return self._scalar_unaop("__neg__")

    def __repr__(self):
        # str() fixes a numpy bug with NaT
        # https://github.com/numpy/numpy/issues/17552
        return (
            f"{self.__class__.__name__}"
            f"({self.value!s}, dtype={self.dtype})"
        )

    def _binop_result_dtype_or_error(self, other, op):
        if op in {"__eq__", "__ne__", "__lt__", "__gt__", "__le__", "__ge__"}:
            return np.bool_

        out_dtype = get_allowed_combinations_for_operator(
            self.dtype, other.dtype, op
        )

        # datetime handling
        if out_dtype in {"M", "m"}:
            if self.dtype.char in {"M", "m"} and other.dtype.char not in {
                "M",
                "m",
            }:
                return self.dtype
            if other.dtype.char in {"M", "m"} and self.dtype.char not in {
                "M",
                "m",
            }:
                return other.dtype
            else:
                if (
                    op == "__sub__"
                    and self.dtype.char == other.dtype.char == "M"
                ):
                    res, _ = np.datetime_data(max(self.dtype, other.dtype))
                    return cudf.dtype("m8" + f"[{res}]")
                return np.result_type(self.dtype, other.dtype)

        return cudf.dtype(out_dtype)

    def _binaryop(self, other, op: str):
        if is_scalar(other):
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
        else:
            return NotImplemented

    def _dispatch_scalar_binop(self, other, op):
        if isinstance(other, Scalar):
            rhs = other.value
        else:
            rhs = other
        lhs = self.value
        reflect, op = self._check_reflected_op(op)
        if reflect:
            lhs, rhs = rhs, lhs
        try:
            return getattr(operator, op)(lhs, rhs)
        except AttributeError:
            return getattr(lhs, op)(rhs)

    def _unaop_result_type_or_error(self, op):
        if op == "__neg__" and self.dtype == "bool":
            raise TypeError(
                "Boolean scalars in cuDF do not support"
                " negation, use logical not"
            )

        if op in {"__ceil__", "__floor__"}:
            if self.dtype.char in "bBhHf?":
                return cudf.dtype("float32")
            else:
                return cudf.dtype("float64")
        return self.dtype

    def _scalar_unaop(self, op):
        out_dtype = self._unaop_result_type_or_error(op)
        if not self.is_valid():
            result = None
        else:
            result = self._dispatch_scalar_unaop(op)
            return Scalar(result, dtype=out_dtype)

    def _dispatch_scalar_unaop(self, op):
        if op == "__floor__":
            return np.floor(self.value)
        if op == "__ceil__":
            return np.ceil(self.value)
        return getattr(self.value, op)()

    def astype(self, dtype):
        if self.dtype == dtype:
            return self
        return Scalar(self.value, dtype)
