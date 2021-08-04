# Copyright (c) 2020-2021, NVIDIA CORPORATION.

import datetime as dt
from collections import namedtuple
from decimal import Decimal

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.dtypes.common import infer_dtype_from_object

import cudf
from cudf.api.types import (  # noqa: F401
    _is_non_decimal_numeric_dtype,
    _is_scalar_or_zero_d_array,
    infer_dtype,
    is_categorical_dtype,
    is_datetime_dtype as is_datetime_dtype,
    is_decimal32_dtype,
    is_decimal64_dtype,
    is_decimal_dtype,
    is_integer,
    is_integer_dtype,
    is_interval_dtype,
    is_list_dtype,
    is_list_like,
    is_numeric_dtype as is_numerical_dtype,
    is_scalar,
    is_string_dtype,
    is_struct_dtype,
    is_timedelta_dtype,
    pandas_dtype,
)
from cudf.core._compat import PANDAS_GE_120

_NA_REP = "<NA>"
_np_pa_dtypes = {
    np.float64: pa.float64(),
    np.float32: pa.float32(),
    np.int64: pa.int64(),
    np.longlong: pa.int64(),
    np.int32: pa.int32(),
    np.int16: pa.int16(),
    np.int8: pa.int8(),
    np.bool_: pa.int8(),
    np.uint64: pa.uint64(),
    np.uint32: pa.uint32(),
    np.uint16: pa.uint16(),
    np.uint8: pa.uint8(),
    np.datetime64: pa.date64(),
    np.object_: pa.string(),
    np.str_: pa.string(),
}

cudf_dtypes_to_pandas_dtypes = {
    np.dtype("uint8"): pd.UInt8Dtype(),
    np.dtype("uint16"): pd.UInt16Dtype(),
    np.dtype("uint32"): pd.UInt32Dtype(),
    np.dtype("uint64"): pd.UInt64Dtype(),
    np.dtype("int8"): pd.Int8Dtype(),
    np.dtype("int16"): pd.Int16Dtype(),
    np.dtype("int32"): pd.Int32Dtype(),
    np.dtype("int64"): pd.Int64Dtype(),
    np.dtype("bool_"): pd.BooleanDtype(),
    np.dtype("object"): pd.StringDtype(),
}

pyarrow_dtypes_to_pandas_dtypes = {
    pa.uint8(): pd.UInt8Dtype(),
    pa.uint16(): pd.UInt16Dtype(),
    pa.uint32(): pd.UInt32Dtype(),
    pa.uint64(): pd.UInt64Dtype(),
    pa.int8(): pd.Int8Dtype(),
    pa.int16(): pd.Int16Dtype(),
    pa.int32(): pd.Int32Dtype(),
    pa.int64(): pd.Int64Dtype(),
    pa.bool_(): pd.BooleanDtype(),
    pa.string(): pd.StringDtype(),
}

pandas_dtypes_to_cudf_dtypes = {
    pd.UInt8Dtype(): np.dtype("uint8"),
    pd.UInt16Dtype(): np.dtype("uint16"),
    pd.UInt32Dtype(): np.dtype("uint32"),
    pd.UInt64Dtype(): np.dtype("uint64"),
    pd.Int8Dtype(): np.dtype("int8"),
    pd.Int16Dtype(): np.dtype("int16"),
    pd.Int32Dtype(): np.dtype("int32"),
    pd.Int64Dtype(): np.dtype("int64"),
    pd.BooleanDtype(): np.dtype("bool_"),
    pd.StringDtype(): np.dtype("object"),
}

if PANDAS_GE_120:
    cudf_dtypes_to_pandas_dtypes[np.dtype("float32")] = pd.Float32Dtype()
    cudf_dtypes_to_pandas_dtypes[np.dtype("float64")] = pd.Float64Dtype()
    pandas_dtypes_to_cudf_dtypes[pd.Float32Dtype()] = np.dtype("float32")
    pandas_dtypes_to_cudf_dtypes[pd.Float64Dtype()] = np.dtype("float64")

SIGNED_INTEGER_TYPES = {"int8", "int16", "int32", "int64"}
UNSIGNED_TYPES = {"uint8", "uint16", "uint32", "uint64"}
INTEGER_TYPES = SIGNED_INTEGER_TYPES | UNSIGNED_TYPES
FLOAT_TYPES = {"float32", "float64"}
SIGNED_TYPES = SIGNED_INTEGER_TYPES | FLOAT_TYPES
NUMERIC_TYPES = SIGNED_TYPES | UNSIGNED_TYPES
DATETIME_TYPES = {
    "datetime64[s]",
    "datetime64[ms]",
    "datetime64[us]",
    "datetime64[ns]",
}
TIMEDELTA_TYPES = {
    "timedelta64[s]",
    "timedelta64[ms]",
    "timedelta64[us]",
    "timedelta64[ns]",
}
OTHER_TYPES = {"bool", "category", "str"}
STRING_TYPES = {"object"}
BOOL_TYPES = {"bool"}
ALL_TYPES = NUMERIC_TYPES | DATETIME_TYPES | TIMEDELTA_TYPES | OTHER_TYPES


def np_to_pa_dtype(dtype):
    """Util to convert numpy dtype to PyArrow dtype.
    """
    # special case when dtype is np.datetime64
    if dtype.kind == "M":
        time_unit, _ = np.datetime_data(dtype)
        if time_unit in ("s", "ms", "us", "ns"):
            # return a pa.Timestamp of the appropriate unit
            return pa.timestamp(time_unit)
        # default is int64_t UNIX ms
        return pa.date64()
    elif dtype.kind == "m":
        time_unit, _ = np.datetime_data(dtype)
        if time_unit in ("s", "ms", "us", "ns"):
            # return a pa.Duration of the appropriate unit
            return pa.duration(time_unit)
        # default fallback unit is ns
        return pa.duration("ns")
    return _np_pa_dtypes[np.dtype(dtype).type]


def get_numeric_type_info(dtype):
    _TypeMinMax = namedtuple("_TypeMinMax", "min,max")
    if dtype.kind in {"i", "u"}:
        info = np.iinfo(dtype)
        return _TypeMinMax(info.min, info.max)
    elif dtype.kind == "f":
        return _TypeMinMax(dtype.type("-inf"), dtype.type("+inf"))
    else:
        raise TypeError(dtype)


def numeric_normalize_types(*args):
    """Cast all args to a common type using numpy promotion logic
    """
    dtype = np.result_type(*[a.dtype for a in args])
    return [a.astype(dtype) for a in args]


def _find_common_type_decimal(dtypes):
    # Find the largest scale and the largest difference between
    # precision and scale of the columns to be concatenated
    s = max([dtype.scale for dtype in dtypes])
    lhs = max([dtype.precision - dtype.scale for dtype in dtypes])
    # Combine to get the necessary precision and clip at the maximum
    # precision
    p = min(cudf.Decimal64Dtype.MAX_PRECISION, s + lhs)
    return cudf.Decimal64Dtype(p, s)


def cudf_dtype_from_pydata_dtype(dtype):
    """ Given a numpy or pandas dtype, converts it into the equivalent cuDF
        Python dtype.
    """

    if is_categorical_dtype(dtype):
        return cudf.core.dtypes.CategoricalDtype
    elif is_decimal32_dtype(dtype):
        return cudf.core.dtypes.Decimal32Dtype
    elif is_decimal64_dtype(dtype):
        return cudf.core.dtypes.Decimal64Dtype
    elif dtype in cudf._lib.types.np_to_cudf_types:
        return dtype.type

    return infer_dtype_from_object(dtype)


def cudf_dtype_to_pa_type(dtype):
    """ Given a cudf pandas dtype, converts it into the equivalent cuDF
        Python dtype.
    """
    if is_categorical_dtype(dtype):
        raise NotImplementedError()
    elif (
        is_list_dtype(dtype)
        or is_struct_dtype(dtype)
        or is_decimal_dtype(dtype)
    ):
        return dtype.to_arrow()
    else:
        return np_to_pa_dtype(np.dtype(dtype))


def cudf_dtype_from_pa_type(typ):
    """ Given a cuDF pyarrow dtype, converts it into the equivalent
        cudf pandas dtype.
    """
    if pa.types.is_list(typ):
        return cudf.core.dtypes.ListDtype.from_arrow(typ)
    elif pa.types.is_struct(typ):
        return cudf.core.dtypes.StructDtype.from_arrow(typ)
    elif pa.types.is_decimal(typ):
        return cudf.core.dtypes.Decimal64Dtype.from_arrow(typ)
    else:
        return pandas_dtype(typ.to_pandas_dtype())


def to_cudf_compatible_scalar(val, dtype=None):
    """
    Converts the value `val` to a numpy/Pandas scalar,
    optionally casting to `dtype`.

    If `val` is None, returns None.
    """

    if cudf._lib.scalar._is_null_host_scalar(val) or isinstance(
        val, cudf.Scalar
    ):
        return val

    if not _is_scalar_or_zero_d_array(val):
        raise ValueError(
            f"Cannot convert value of type {type(val).__name__} "
            "to cudf scalar"
        )

    if isinstance(val, Decimal):
        return val

    if isinstance(val, (np.ndarray, cp.ndarray)) and val.ndim == 0:
        val = val.item()

    if ((dtype is None) and isinstance(val, str)) or is_string_dtype(dtype):
        dtype = "str"

    if isinstance(val, dt.datetime):
        val = np.datetime64(val)
    elif isinstance(val, dt.timedelta):
        val = np.timedelta64(val)
    elif isinstance(val, pd.Timestamp):
        val = val.to_datetime64()
    elif isinstance(val, pd.Timedelta):
        val = val.to_timedelta64()

    val = pandas_dtype(type(val)).type(val)

    if dtype is not None:
        val = val.astype(dtype)

    if val.dtype.type is np.datetime64:
        time_unit, _ = np.datetime_data(val.dtype)
        if time_unit in ("D", "W", "M", "Y"):
            val = val.astype("datetime64[s]")
    elif val.dtype.type is np.timedelta64:
        time_unit, _ = np.datetime_data(val.dtype)
        if time_unit in ("D", "W", "M", "Y"):
            val = val.astype("timedelta64[ns]")

    return val


def is_column_like(obj):
    """
    This function checks if the given `obj`
    is a column-like (Series, Index...)
    type or not.

    Parameters
    ----------
    obj : object of any type which needs to be validated.

    Returns
    -------
    Boolean: True or False depending on whether the
    input `obj` is column-like or not.
    """
    return (
        isinstance(
            obj,
            (
                cudf.core.column.ColumnBase,
                cudf.Series,
                cudf.Index,
                pd.Series,
                pd.Index,
            ),
        )
        or (
            hasattr(obj, "__cuda_array_interface__")
            and len(obj.__cuda_array_interface__["shape"]) == 1
        )
        or (
            hasattr(obj, "__array_interface__")
            and len(obj.__array_interface__["shape"]) == 1
        )
    )


def can_convert_to_column(obj):
    """
    This function checks if the given `obj`
    can be used to create a column or not.

    Parameters
    ----------
    obj : object of any type which needs to be validated.

    Returns
    -------
    Boolean: True or False depending on whether the
    input `obj` is column-compatible or not.
    """
    return is_column_like(obj) or is_list_like(obj)


def min_scalar_type(a, min_size=8):
    return min_signed_type(a, min_size=min_size)


def min_signed_type(x, min_size=8):
    """
    Return the smallest *signed* integer dtype
    that can represent the integer ``x``
    """
    for int_dtype in np.sctypes["int"]:
        if (np.dtype(int_dtype).itemsize * 8) >= min_size:
            if np.iinfo(int_dtype).min <= x <= np.iinfo(int_dtype).max:
                return int_dtype
    # resort to using `int64` and let numpy raise appropriate exception:
    return np.int64(x).dtype


def min_unsigned_type(x, min_size=8):
    """
    Return the smallest *unsigned* integer dtype
    that can represent the integer ``x``
    """
    for int_dtype in np.sctypes["uint"]:
        if (np.dtype(int_dtype).itemsize * 8) >= min_size:
            if 0 <= x <= np.iinfo(int_dtype).max:
                return int_dtype
    # resort to using `uint64` and let numpy raise appropriate exception:
    return np.uint64(x).dtype


def min_column_type(x, expected_type):
    """
    Return the smallest dtype which can represent all
    elements of the `NumericalColumn` `x`
    If the column is not a subtype of `np.signedinteger` or `np.floating`
    returns the same dtype as the dtype of `x` without modification
    """

    if not isinstance(x, cudf.core.column.NumericalColumn):
        raise TypeError("Argument x must be of type column.NumericalColumn")
    if x.valid_count == 0:
        return x.dtype

    if np.issubdtype(x.dtype, np.floating):
        max_bound_dtype = np.min_scalar_type(x.max())
        min_bound_dtype = np.min_scalar_type(x.min())
        result_type = np.promote_types(max_bound_dtype, min_bound_dtype)
        if result_type == np.dtype("float16"):
            # cuDF does not support float16 dtype
            result_type = np.dtype("float32")
        return result_type

    if np.issubdtype(expected_type, np.integer):
        max_bound_dtype = np.min_scalar_type(x.max())
        min_bound_dtype = np.min_scalar_type(x.min())
        return np.promote_types(max_bound_dtype, min_bound_dtype)

    return x.dtype


def get_min_float_dtype(col):
    max_bound_dtype = np.min_scalar_type(float(col.max()))
    min_bound_dtype = np.min_scalar_type(float(col.min()))
    result_type = np.promote_types(max_bound_dtype, min_bound_dtype)
    if result_type == np.dtype("float16"):
        # cuDF does not support float16 dtype
        result_type = np.dtype("float32")
    return result_type


def check_cast_unsupported_dtype(dtype):
    if is_categorical_dtype(dtype):
        return dtype

    if isinstance(dtype, pd.core.arrays.numpy_.PandasDtype):
        dtype = dtype.numpy_dtype
    else:
        dtype = np.dtype(dtype)

    if dtype in cudf._lib.types.np_to_cudf_types:
        return dtype

    if dtype == np.dtype("float16"):
        return np.dtype("float32")

    raise NotImplementedError(
        f"Cannot cast {dtype} dtype, as it is not supported by CuDF."
    )


def is_mixed_with_object_dtype(lhs, rhs):
    return (lhs.dtype == "object" and rhs.dtype != "object") or (
        rhs.dtype == "object" and lhs.dtype != "object"
    )


def get_time_unit(obj):
    if isinstance(
        obj,
        (
            cudf.core.column.datetime.DatetimeColumn,
            cudf.core.column.timedelta.TimeDeltaColumn,
        ),
    ):
        return obj.time_unit

    time_unit, _ = np.datetime_data(obj.dtype)
    return time_unit


def _get_nan_for_dtype(dtype):
    dtype = np.dtype(dtype)
    if pd.api.types.is_datetime64_dtype(
        dtype
    ) or pd.api.types.is_timedelta64_dtype(dtype):
        time_unit, _ = np.datetime_data(dtype)
        return dtype.type("nat", time_unit)
    elif dtype.kind == "f":
        return dtype.type("nan")
    else:
        return np.float64("nan")


def _decimal_to_int64(decimal: Decimal) -> int:
    """
    Scale a Decimal such that the result is the integer
    that would result from removing the decimal point.

    Examples
    --------
    >>> _decimal_to_int64(Decimal('1.42'))
    142
    >>> _decimal_to_int64(Decimal('0.0042'))
    42
    >>> _decimal_to_int64(Decimal('-1.004201'))
    -1004201

    """
    return int(f"{decimal:0f}".replace(".", ""))


def get_allowed_combinations_for_operator(dtype_l, dtype_r, op):
    error = TypeError(
        f"{op} not supported between {dtype_l} and {dtype_r} scalars"
    )

    to_numpy_ops = {
        "__add__": _ADD_TYPES,
        "__sub__": _SUB_TYPES,
        "__mul__": _MUL_TYPES,
        "__floordiv__": _FLOORDIV_TYPES,
        "__truediv__": _TRUEDIV_TYPES,
        "__mod__": _MOD_TYPES,
        "__pow__": _POW_TYPES,
    }
    allowed = to_numpy_ops.get(op, op)

    # special rules for string
    if dtype_l == "object" or dtype_r == "object":
        if (dtype_l == dtype_r == "object") and op == "__add__":
            return "str"
        else:
            raise error

    # Check if we can directly operate

    for valid_combo in allowed:
        ltype, rtype, outtype = valid_combo
        if np.can_cast(dtype_l.char, ltype) and np.can_cast(
            dtype_r.char, rtype
        ):
            return outtype

    raise error


def find_common_type(dtypes):
    """
    Wrapper over np.find_common_type to handle special cases

    Corner cases:
    1. "M8", "M8" -> "M8" | "m8", "m8" -> "m8"

    Parameters
    ----------
    dtypes : iterable, sequence of dtypes to find common types

    Returns
    -------
    dtype : np.dtype optional, the result from np.find_common_type,
    None if input is empty

    """

    if len(dtypes) == 0:
        return None

    # Aggregate same types
    dtypes = set(dtypes)

    if any(is_decimal_dtype(dtype) for dtype in dtypes):
        if all(
            is_decimal_dtype(dtype) or is_numerical_dtype(dtype)
            for dtype in dtypes
        ):
            return _find_common_type_decimal(
                [dtype for dtype in dtypes if is_decimal_dtype(dtype)]
            )
        else:
            return np.dtype("O")

    # Corner case 1:
    # Resort to np.result_type to handle "M" and "m" types separately
    dt_dtypes = set(filter(lambda t: is_datetime_dtype(t), dtypes))
    if len(dt_dtypes) > 0:
        dtypes = dtypes - dt_dtypes
        dtypes.add(np.result_type(*dt_dtypes))

    td_dtypes = set(
        filter(lambda t: pd.api.types.is_timedelta64_dtype(t), dtypes)
    )
    if len(td_dtypes) > 0:
        dtypes = dtypes - td_dtypes
        dtypes.add(np.result_type(*td_dtypes))

    common_dtype = np.find_common_type(list(dtypes), [])
    if common_dtype == np.dtype("float16"):
        # cuDF does not support float16 dtype
        return np.dtype("float32")
    else:
        return common_dtype


def _can_cast(from_dtype, to_dtype):
    """
    Utility function to determine if we can cast
    from `from_dtype` to `to_dtype`. This function primarily calls
    `np.can_cast` but with some special handling around
    cudf specific dtypes.
    """
    if isinstance(from_dtype, type):
        from_dtype = np.dtype(from_dtype)
    if isinstance(to_dtype, type):
        to_dtype = np.dtype(to_dtype)

    # TODO : Add precision & scale checking for
    # decimal types in future
    if isinstance(from_dtype, cudf.core.dtypes.Decimal64Dtype):
        if isinstance(to_dtype, cudf.core.dtypes.Decimal64Dtype):
            return True
        elif isinstance(to_dtype, np.dtype):
            if to_dtype.kind in {"i", "f", "u", "U", "O"}:
                return True
            else:
                return False
    elif isinstance(from_dtype, np.dtype):
        if isinstance(to_dtype, np.dtype):
            return np.can_cast(from_dtype, to_dtype)
        elif isinstance(to_dtype, cudf.core.dtypes.Decimal64Dtype):
            if from_dtype.kind in {"i", "f", "u", "U", "O"}:
                return True
            else:
                return False
        elif isinstance(to_dtype, cudf.core.types.CategoricalDtype):
            return True
        else:
            return False
    elif isinstance(from_dtype, cudf.core.dtypes.ListDtype):
        # TODO: Add level based checks too once casting of
        # list columns is supported
        if isinstance(to_dtype, cudf.core.dtypes.ListDtype):
            return np.can_cast(from_dtype.leaf_type, to_dtype.leaf_type)
        else:
            return False
    elif isinstance(from_dtype, cudf.core.dtypes.CategoricalDtype):
        if isinstance(to_dtype, cudf.core.dtypes.CategoricalDtype):
            return True
        elif isinstance(to_dtype, np.dtype):
            return np.can_cast(from_dtype._categories.dtype, to_dtype)
        else:
            return False
    else:
        return np.can_cast(from_dtype, to_dtype)


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
