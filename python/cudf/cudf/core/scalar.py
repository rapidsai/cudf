# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from __future__ import annotations

import copy
import datetime
import decimal
import functools
import operator
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf.api.types import is_scalar
from cudf.core.dtypes import (
    Decimal32Dtype,
    Decimal64Dtype,
    ListDtype,
    StructDtype,
    _BaseDtype,
)
from cudf.core.missing import NA, NaT
from cudf.core.mixins import BinaryOperand
from cudf.utils.dtypes import (
    CUDF_STRING_DTYPE,
    _maybe_convert_to_default_type,
    cudf_dtype_from_pa_type,
    find_common_type,
)

if TYPE_CHECKING:
    from typing_extensions import Self

    from cudf._typing import Dtype, ScalarLike


# Type dispatch loops similar to what are found in `np.add.types`
# In NumPy, whether or not an op can be performed between two
# operands is determined by checking to see if NumPy has a c/c++
# loop specifically for adding those two operands built in. If
# not it will search lists like these for a loop for types that
# the operands can be safely cast to. These are those lookups,
# modified slightly for cuDF's rules
_ADD_TYPES = [
    "???",
    "BBB",
    "HHH",
    "III",
    "LLL",
    "bbb",
    "hhh",
    "iii",
    "lll",
    "fff",
    "ddd",
    "mMM",
    "MmM",
    "mmm",
    "LMM",
    "MLM",
    "Lmm",
    "mLm",
]
_SUB_TYPES = [
    "BBB",
    "HHH",
    "III",
    "LLL",
    "bbb",
    "hhh",
    "iii",
    "lll",
    "fff",
    "ddd",
    "???",
    "MMm",
    "mmm",
    "MmM",
    "MLM",
    "mLm",
    "Lmm",
]
_MUL_TYPES = [
    "???",
    "BBB",
    "HHH",
    "III",
    "LLL",
    "bbb",
    "hhh",
    "iii",
    "lll",
    "fff",
    "ddd",
    "mLm",
    "Lmm",
    "mlm",
    "lmm",
]
_FLOORDIV_TYPES = [
    "bbb",
    "BBB",
    "HHH",
    "III",
    "LLL",
    "hhh",
    "iii",
    "lll",
    "fff",
    "ddd",
    "???",
    "mqm",
    "mdm",
    "mmq",
]
_TRUEDIV_TYPES = ["fff", "ddd", "mqm", "mmd", "mLm"]
_MOD_TYPES = [
    "bbb",
    "BBB",
    "hhh",
    "HHH",
    "iii",
    "III",
    "lll",
    "LLL",
    "fff",
    "ddd",
    "mmm",
]
_POW_TYPES = [
    "bbb",
    "BBB",
    "hhh",
    "HHH",
    "iii",
    "III",
    "lll",
    "LLL",
    "fff",
    "ddd",
]


def to_cudf_compatible_scalar(val, dtype=None):
    """
    Converts the value `val` to a numpy/Pandas scalar,
    optionally casting to `dtype`.

    If `val` is None, returns None.
    """

    if cudf.utils.utils._is_null_host_scalar(val) or isinstance(
        val, cudf.Scalar
    ):
        return val

    if not cudf.api.types._is_scalar_or_zero_d_array(val):
        raise ValueError(
            f"Cannot convert value of type {type(val).__name__} to cudf scalar"
        )

    if isinstance(val, decimal.Decimal):
        return val

    if isinstance(val, (np.ndarray, cp.ndarray)) and val.ndim == 0:
        val = val.item()

    if (
        (dtype is None) and isinstance(val, str)
    ) or cudf.api.types.is_string_dtype(dtype):
        dtype = "str"

        if isinstance(val, str) and val.endswith("\x00"):
            # Numpy string dtypes are fixed width and use NULL to
            # indicate the end of the string, so they cannot
            # distinguish between "abc\x00" and "abc".
            # https://github.com/numpy/numpy/issues/20118
            # In this case, don't try going through numpy and just use
            # the string value directly (cudf.DeviceScalar will DTRT)
            return val

    tz_error_msg = (
        "Cannot covert a timezone-aware timestamp to timezone-naive scalar."
    )
    if isinstance(val, pd.Timestamp):
        if val.tz is not None:
            raise NotImplementedError(tz_error_msg)

        val = val.to_datetime64()
    elif isinstance(val, pd.Timedelta):
        val = val.to_timedelta64()
    elif isinstance(val, datetime.datetime):
        if val.tzinfo is not None:
            raise NotImplementedError(tz_error_msg)
        val = np.datetime64(val)
    elif isinstance(val, datetime.timedelta):
        val = np.timedelta64(val)

    if dtype is not None:
        dtype = np.dtype(dtype)
        if isinstance(val, str) and dtype.kind == "M":
            # pd.Timestamp can handle str, but not np.str_
            val = pd.Timestamp(str(val)).to_datetime64().astype(dtype)
        else:
            # At least datetimes cannot be converted to scalar via dtype.type:
            val = np.array(val, dtype)[()]
    else:
        val = _maybe_convert_to_default_type(
            cudf.api.types.pandas_dtype(type(val))
        ).type(val)

    if val.dtype.type is np.datetime64:
        time_unit, _ = np.datetime_data(val.dtype)
        if time_unit in ("D", "W", "M", "Y"):
            val = val.astype("datetime64[s]")
    elif val.dtype.type is np.timedelta64:
        time_unit, _ = np.datetime_data(val.dtype)
        if time_unit in ("D", "W", "M", "Y"):
            val = val.astype("timedelta64[ns]")

    return val


def get_allowed_combinations_for_operator(
    dtype_l: np.dtype, dtype_r: np.dtype, op: str
) -> np.dtype:
    error = TypeError(
        f"{op} not supported between {dtype_l} and {dtype_r} scalars"
    )

    to_numpy_ops = {
        "__add__": _ADD_TYPES,
        "__radd__": _ADD_TYPES,
        "__sub__": _SUB_TYPES,
        "__rsub__": _SUB_TYPES,
        "__mul__": _MUL_TYPES,
        "__rmul__": _MUL_TYPES,
        "__floordiv__": _FLOORDIV_TYPES,
        "__rfloordiv__": _FLOORDIV_TYPES,
        "__truediv__": _TRUEDIV_TYPES,
        "__rtruediv__": _TRUEDIV_TYPES,
        "__mod__": _MOD_TYPES,
        "__rmod__": _MOD_TYPES,
        "__pow__": _POW_TYPES,
        "__rpow__": _POW_TYPES,
    }
    allowed = to_numpy_ops.get(op, op)

    # special rules for string
    if dtype_l == "object" or dtype_r == "object":
        if (dtype_l == dtype_r == "object") and op == "__add__":
            return CUDF_STRING_DTYPE
        else:
            raise error

    # Check if we can directly operate

    for valid_combo in allowed:
        ltype, rtype, outtype = valid_combo  # type: ignore[misc]
        if np.can_cast(dtype_l.char, ltype) and np.can_cast(  # type: ignore[has-type]  # noqa: TID251
            dtype_r.char,
            rtype,  # type: ignore[has-type]
        ):
            return np.dtype(outtype)  # type: ignore[has-type]

    raise error


def _preprocess_host_value(value, dtype) -> tuple[ScalarLike, Dtype]:
    """
    Preprocess a value and dtype for host-side cudf.Scalar

    Parameters
    ----------
    value: Scalarlike
    dtype: dtypelike or None

    Returns
    -------
    tuple[ScalarLike, Dtype]
    """
    valid = not cudf.utils.utils._is_null_host_scalar(value)

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

    if isinstance(value, pa.Scalar):
        # TODO: Avoid converting to a Python scalar since we
        # end up converting pyarrow.Scalars to pylibcudf.Scalars
        if dtype is None:
            dtype = cudf_dtype_from_pa_type(value.type)
        return value.as_py(), dtype

    if isinstance(dtype, cudf.core.dtypes.DecimalDtype):
        if isinstance(value, np.integer):
            value = int(value)
        value = pa.scalar(value, type=dtype.to_arrow()).as_py()
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
                    raise TypeError("Cant convert generic NaT to null scalar")
                else:
                    dtype = value.dtype
            else:
                raise TypeError(
                    "dtype required when constructing a null scalar"
                )
        else:
            dtype = value.dtype

    if not isinstance(dtype, (np.dtype, _BaseDtype)):
        dtype = cudf.dtype(dtype)
    elif dtype.kind == "U":
        dtype = CUDF_STRING_DTYPE

    if not valid:
        value = NaT if dtype.kind in "mM" else NA

    return value, dtype


def _replace_nested(obj, check, replacement):
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if check(item):
                obj[i] = replacement
            elif isinstance(item, (dict, list)):
                _replace_nested(item, check, replacement)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if check(v):
                obj[k] = replacement
            elif isinstance(v, (dict, list)):
                _replace_nested(v, check, replacement)


def _maybe_nested_pa_scalar_to_py(pa_scalar: pa.Scalar) -> Any:
    """
    Convert a "nested" pyarrow scalar to a Python object.

    These scalars come from pylibcudf.Scalar where field names can be
    duplicate empty strings.

    Parameters
    ----------
    pa_scalar: pa.Scalar

    Returns
    -------
    Any
        Python scalar
    """
    if not pa_scalar.is_valid:
        return pa_scalar.as_py()
    elif pa.types.is_struct(pa_scalar.type):
        return {
            str(i): _maybe_nested_pa_scalar_to_py(val)
            for i, (_, val) in enumerate(pa_scalar.items())
        }
    elif pa.types.is_list(pa_scalar.type):
        return [_maybe_nested_pa_scalar_to_py(val) for val in pa_scalar]
    else:
        return pa_scalar.as_py()


def _to_plc_scalar(value: ScalarLike, dtype: Dtype) -> plc.Scalar:
    """
    Convert a value and dtype to a pylibcudf Scalar for device-side cudf.Scalar

    Parameters
    ----------
    value: Scalarlike
    dtype: dtypelike

    Returns
    -------
    pylibcudf.Scalar
        pylibcudf.Scalar for cudf.Scalar._device_value
    """
    if cudf.utils.utils.is_na_like(value):
        value = None
    else:
        # TODO: For now we deepcopy the input value for nested values to avoid
        # overwriting the input values when replacing nulls. Since it's
        # just host values it's not that expensive, but we could consider
        # alternatives.
        if isinstance(value, (list, dict)):
            value = copy.deepcopy(value)
        _replace_nested(value, cudf.utils.utils.is_na_like, None)

    if isinstance(dtype, cudf.core.dtypes._BaseDtype):
        pa_type = dtype.to_arrow()
    elif dtype == CUDF_STRING_DTYPE:
        # Have to manually convert object types, which we use internally
        # for strings but pyarrow only supports as unicode 'U'
        pa_type = pa.string()
    else:
        pa_type = pa.from_numpy_dtype(dtype)

    pa_scalar = pa.scalar(value, type=pa_type)
    plc_scalar = plc.interop.from_arrow(pa_scalar)
    if isinstance(dtype, (Decimal32Dtype, Decimal64Dtype)):
        # pyarrow only supports decimal128
        if isinstance(dtype, Decimal32Dtype):
            plc_type = plc.DataType(plc.TypeId.DECIMAL32, -dtype.scale)
        elif isinstance(dtype, Decimal64Dtype):
            plc_type = plc.DataType(plc.TypeId.DECIMAL64, -dtype.scale)
        plc_column = plc.unary.cast(
            plc.Column.from_scalar(plc_scalar, 1), plc_type
        )
        plc_scalar = plc.copying.get_element(plc_column, 0)
    return plc_scalar


@functools.lru_cache(maxsize=128)
def pa_scalar_to_plc_scalar(pa_scalar: pa.Scalar) -> plc.Scalar:
    """
    Cached conversion from a pyarrow.Scalar to pylibcudf.Scalar.

    Intended to replace CachedScalarInstanceMeta in the future.

    Parameters
    ----------
    pa_scalar: pa.Scalar

    Returns
    -------
    pylibcudf.Scalar
        pylibcudf.Scalar to use in pylibcudf APIs
    """
    return plc.interop.from_arrow(pa_scalar)


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

    Parameters
    ----------
    value : Python Scalar, NumPy Scalar, or cuDF Scalar
        The scalar value to be converted to a GPU backed scalar object
    dtype : np.dtype or string specifier
        The data type
    """

    _VALID_BINARY_OPERATIONS = BinaryOperand._SUPPORTED_BINARY_OPERATIONS

    def __init__(self, value, dtype=None):
        warnings.warn(
            "Scalar is deprecated and will be removed in 25.08.", FutureWarning
        )
        self._host_value = None
        self._host_dtype = None
        self._device_value: plc.Scalar | None = None

        if isinstance(value, Scalar):
            if value._is_host_value_current:
                self._host_value = value._host_value
                self._host_dtype = value._host_dtype
            else:
                self._device_value = value._device_value
        else:
            self._host_value, self._host_dtype = _preprocess_host_value(
                value, dtype
            )

    @classmethod
    def from_pylibcudf(cls, scalar: plc.Scalar) -> Self:
        if not isinstance(scalar, plc.Scalar):
            raise TypeError(
                "Expected an instance of pylibcudf.Scalar, "
                f"got {type(scalar).__name__}"
            )
        obj = object.__new__(cls)
        obj._host_value = None
        obj._host_dtype = None
        obj._device_value = scalar
        return obj

    @property
    def _is_host_value_current(self) -> bool:
        return self._host_value is not None

    @property
    def _is_device_value_current(self) -> bool:
        return self._device_value is not None

    @property
    def device_value(self) -> plc.Scalar:
        self._sync()
        return self._device_value

    @property
    def value(self):
        if not self._is_host_value_current:
            self._device_value_to_host()
        return self._host_value

    # TODO: change to @functools.cached_property
    @property
    def dtype(self):
        if self._host_dtype is not None:
            return self._host_dtype
        if not self._is_host_value_current:
            self._device_value_to_host()
        _, host_dtype = _preprocess_host_value(self._host_value, None)
        self._host_dtype = host_dtype
        return self._host_dtype

    def is_valid(self) -> bool:
        if not self._is_host_value_current:
            self._device_value_to_host()
        return not cudf.utils.utils._is_null_host_scalar(self._host_value)

    def _device_value_to_host(self) -> None:
        ps = plc.interop.to_arrow(self._device_value)
        is_datetime = pa.types.is_timestamp(ps.type)
        is_timedelta = pa.types.is_duration(ps.type)
        if not ps.is_valid:
            if is_datetime or is_timedelta:
                self._host_value = NaT
            else:
                self._host_value = NA
        else:
            # TODO: The special handling of specific types below does not currently
            # extend to nested types containing those types (e.g. List[timedelta]
            # where the timedelta would overflow). We should eventually account for
            # those cases, but that will require more careful consideration of how
            # to traverse the contents of the nested data.
            if is_datetime or is_timedelta:
                time_unit = ps.type.unit
                # Cast to int64 to avoid overflow
                ps_cast = ps.cast(pa.int64()).as_py()
                out_type = np.datetime64 if is_datetime else np.timedelta64
                self._host_value = out_type(ps_cast, time_unit)
            elif (
                pa.types.is_integer(ps.type)
                or pa.types.is_floating(ps.type)
                or pa.types.is_boolean(ps.type)
            ):
                self._host_value = ps.type.to_pandas_dtype()(ps.as_py())
            else:
                host_value = _maybe_nested_pa_scalar_to_py(ps)
                _replace_nested(host_value, lambda item: item is None, NA)
                self._host_value = host_value

    def _sync(self) -> None:
        """
        If the cache is not synched, copy either the device or host value
        to the host or device respectively. If cache is valid, do nothing
        """
        if self._is_host_value_current and self._is_device_value_current:
            return
        elif self._is_host_value_current and not self._is_device_value_current:
            self._device_value = _to_plc_scalar(
                self._host_value, self._host_dtype
            )
        elif self._is_device_value_current and not self._is_host_value_current:
            self._device_value_to_host()
            self._host_dtype = self._host_value.dtype
        else:
            raise ValueError("Invalid cudf.Scalar")

    def __index__(self) -> int:
        if self.dtype.kind not in {"u", "i"}:
            raise TypeError("Only Integer typed scalars may be used in slices")
        return int(self)

    def __int__(self) -> int:
        return int(self.value)

    def __float__(self) -> float:
        return float(self.value)

    def __bool__(self) -> bool:
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

    def __repr__(self) -> str:
        # str() fixes a numpy bug with NaT
        # https://github.com/numpy/numpy/issues/17552
        return f"{self.__class__.__name__}({self.value!s}, dtype={self.dtype})"

    def _binop_result_dtype_or_error(self, other, op) -> np.dtype:
        if op in {"__eq__", "__ne__", "__lt__", "__gt__", "__le__", "__ge__"}:
            return np.dtype(np.bool_)

        out_dtype = get_allowed_combinations_for_operator(
            self.dtype, other.dtype, op
        )

        # datetime handling
        if out_dtype.kind in {"M", "m"}:
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
                    return np.dtype(f"m8[{res}]")
                return find_common_type((self.dtype, other.dtype))

        return out_dtype

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
                return np.dtype(np.float32)
            else:
                return np.dtype(np.float64)
        return self.dtype

    def _scalar_unaop(self, op) -> None | Self:
        out_dtype = self._unaop_result_type_or_error(op)
        if not self.is_valid():
            return None
        else:
            result = self._dispatch_scalar_unaop(op)
            return Scalar(result, dtype=out_dtype)  # type: ignore[return-value]

    def _dispatch_scalar_unaop(self, op):
        if op == "__floor__":
            return np.floor(self.value)
        if op == "__ceil__":
            return np.ceil(self.value)
        return getattr(self.value, op)()

    def astype(self, dtype) -> Self:
        if self.dtype == dtype:
            return self
        return Scalar(self.value, dtype)  # type: ignore[return-value]
