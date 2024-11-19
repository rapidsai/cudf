# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from __future__ import annotations

import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.dtypes.common import infer_dtype_from_object

import cudf

if TYPE_CHECKING:
    from cudf._typing import DtypeObj

"""Map numpy dtype to pyarrow types.
Note that np.bool_ bitwidth (8) is different from pa.bool_ (1). Special
handling is required when converting a Boolean column into arrow.
"""
_np_pa_dtypes = {
    np.float64: pa.float64(),
    np.float32: pa.float32(),
    np.int64: pa.int64(),
    np.longlong: pa.int64(),
    np.int32: pa.int32(),
    np.int16: pa.int16(),
    np.int8: pa.int8(),
    np.bool_: pa.bool_(),
    np.uint64: pa.uint64(),
    np.uint32: pa.uint32(),
    np.uint16: pa.uint16(),
    np.uint8: pa.uint8(),
    np.datetime64: pa.date64(),
    np.object_: pa.string(),
    np.str_: pa.string(),
}

np_dtypes_to_pandas_dtypes = {
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
    np.dtype("float32"): pd.Float32Dtype(),
    np.dtype("float64"): pd.Float64Dtype(),
}
pandas_dtypes_to_np_dtypes = {
    pd_dtype: np_dtype
    for np_dtype, pd_dtype in np_dtypes_to_pandas_dtypes.items()
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
    """Util to convert numpy dtype to PyArrow dtype."""
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
    return _np_pa_dtypes[cudf.dtype(dtype).type]


def _find_common_type_decimal(dtypes):
    # Find the largest scale and the largest difference between
    # precision and scale of the columns to be concatenated
    s = max(dtype.scale for dtype in dtypes)
    lhs = max(dtype.precision - dtype.scale for dtype in dtypes)
    # Combine to get the necessary precision and clip at the maximum
    # precision
    p = s + lhs

    if p > cudf.Decimal64Dtype.MAX_PRECISION:
        return cudf.Decimal128Dtype(
            min(cudf.Decimal128Dtype.MAX_PRECISION, p), s
        )
    elif p > cudf.Decimal32Dtype.MAX_PRECISION:
        return cudf.Decimal64Dtype(
            min(cudf.Decimal64Dtype.MAX_PRECISION, p), s
        )
    else:
        return cudf.Decimal32Dtype(
            min(cudf.Decimal32Dtype.MAX_PRECISION, p), s
        )


def cudf_dtype_from_pydata_dtype(dtype):
    """Given a numpy or pandas dtype, converts it into the equivalent cuDF
    Python dtype.
    """

    if cudf.api.types._is_categorical_dtype(dtype):
        return cudf.core.dtypes.CategoricalDtype
    elif cudf.api.types.is_decimal32_dtype(dtype):
        return cudf.core.dtypes.Decimal32Dtype
    elif cudf.api.types.is_decimal64_dtype(dtype):
        return cudf.core.dtypes.Decimal64Dtype
    elif cudf.api.types.is_decimal128_dtype(dtype):
        return cudf.core.dtypes.Decimal128Dtype
    elif dtype in cudf._lib.types.SUPPORTED_NUMPY_TO_LIBCUDF_TYPES:
        return dtype.type

    return infer_dtype_from_object(dtype)


def cudf_dtype_to_pa_type(dtype):
    """Given a cudf pandas dtype, converts it into the equivalent cuDF
    Python dtype.
    """
    if isinstance(dtype, cudf.CategoricalDtype):
        raise NotImplementedError(
            "No conversion from Categorical to pyarrow type"
        )
    elif isinstance(
        dtype,
        (cudf.StructDtype, cudf.ListDtype, cudf.core.dtypes.DecimalDtype),
    ):
        return dtype.to_arrow()
    else:
        return np_to_pa_dtype(cudf.dtype(dtype))


def cudf_dtype_from_pa_type(typ):
    """Given a cuDF pyarrow dtype, converts it into the equivalent
    cudf pandas dtype.
    """
    if pa.types.is_list(typ):
        return cudf.core.dtypes.ListDtype.from_arrow(typ)
    elif pa.types.is_struct(typ):
        return cudf.core.dtypes.StructDtype.from_arrow(typ)
    elif pa.types.is_decimal(typ):
        return cudf.core.dtypes.Decimal128Dtype.from_arrow(typ)
    elif pa.types.is_large_string(typ):
        return cudf.dtype("str")
    else:
        return cudf.api.types.pandas_dtype(typ.to_pandas_dtype())


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

    if not cudf.api.types._is_scalar_or_zero_d_array(val):
        raise ValueError(
            f"Cannot convert value of type {type(val).__name__} "
            "to cudf scalar"
        )

    if isinstance(val, Decimal):
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
    return is_column_like(obj) or cudf.api.types.is_list_like(obj)


def min_signed_type(x: int, min_size: int = 8) -> np.dtype:
    """
    Return the smallest *signed* integer dtype
    that can represent the integer ``x``
    """
    for int_dtype in (np.int8, np.int16, np.int32, np.int64):
        if (cudf.dtype(int_dtype).itemsize * 8) >= min_size:
            if np.iinfo(int_dtype).min <= x <= np.iinfo(int_dtype).max:
                return np.dtype(int_dtype)
    # resort to using `int64` and let numpy raise appropriate exception:
    return np.int64(x).dtype


def min_unsigned_type(x: int, min_size: int = 8) -> np.dtype:
    """
    Return the smallest *unsigned* integer dtype
    that can represent the integer ``x``
    """
    for int_dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if (cudf.dtype(int_dtype).itemsize * 8) >= min_size:
            if 0 <= x <= np.iinfo(int_dtype).max:
                return np.dtype(int_dtype)
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
    if x.null_count == len(x):
        return x.dtype

    min_value, max_value = x.min(), x.max()
    either_is_inf = np.isinf(min_value) or np.isinf(max_value)
    expected_type = cudf.dtype(expected_type)
    if not either_is_inf and expected_type.kind in "i":
        max_bound_dtype = min_signed_type(max_value)
        min_bound_dtype = min_signed_type(min_value)
        result_type = np.promote_types(max_bound_dtype, min_bound_dtype)
    elif not either_is_inf and expected_type.kind in "u":
        max_bound_dtype = min_unsigned_type(max_value)
        min_bound_dtype = min_unsigned_type(min_value)
        result_type = np.promote_types(max_bound_dtype, min_bound_dtype)
    elif x.dtype.kind == "f":
        return get_min_float_dtype(x)
    else:
        result_type = x.dtype

    return cudf.dtype(result_type)


def get_min_float_dtype(col):
    max_bound_dtype = np.min_scalar_type(float(col.max()))
    min_bound_dtype = np.min_scalar_type(float(col.min()))
    result_type = np.promote_types(
        "float32", np.promote_types(max_bound_dtype, min_bound_dtype)
    )
    return cudf.dtype(result_type)


def is_mixed_with_object_dtype(lhs, rhs):
    if isinstance(lhs.dtype, cudf.CategoricalDtype):
        return is_mixed_with_object_dtype(lhs.dtype.categories, rhs)
    elif isinstance(rhs.dtype, cudf.CategoricalDtype):
        return is_mixed_with_object_dtype(lhs, rhs.dtype.categories)

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
    dtype = cudf.dtype(dtype)
    if dtype.kind in "mM":
        time_unit, _ = np.datetime_data(dtype)
        return dtype.type("nat", time_unit)
    elif dtype.kind == "f":
        return dtype.type("nan")
    else:
        return np.float64("nan")


def get_allowed_combinations_for_operator(dtype_l, dtype_r, op):
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

    # Early exit for categoricals since they're not hashable and therefore
    # can't be put in a set.
    if any(cudf.api.types._is_categorical_dtype(dtype) for dtype in dtypes):
        if all(
            (
                cudf.api.types._is_categorical_dtype(dtype)
                and (not dtype.ordered if hasattr(dtype, "ordered") else True)
            )
            for dtype in dtypes
        ):
            if len({dtype._categories.dtype for dtype in dtypes}) == 1:
                return cudf.CategoricalDtype(
                    cudf.core.column.concat_columns(
                        [dtype._categories for dtype in dtypes]
                    ).unique()
                )
            else:
                raise ValueError(
                    "Only unordered categories of the same underlying type "
                    "may be coerced to a common type."
                )
        else:
            # TODO: Should this be an error case (mixing categorical with other
            # dtypes) or should this return object? Unclear if we have enough
            # information to decide right now, may have to come back to this as
            # usage of find_common_type increases.
            return cudf.dtype("O")

    # Aggregate same types
    dtypes = {cudf.dtype(dtype) for dtype in dtypes}
    if len(dtypes) == 1:
        return dtypes.pop()

    if any(
        isinstance(dtype, cudf.core.dtypes.DecimalDtype) for dtype in dtypes
    ):
        if all(cudf.api.types.is_numeric_dtype(dtype) for dtype in dtypes):
            return _find_common_type_decimal(
                [
                    dtype
                    for dtype in dtypes
                    if cudf.api.types.is_decimal_dtype(dtype)
                ]
            )
        else:
            return cudf.dtype("O")
    elif any(
        isinstance(dtype, (cudf.ListDtype, cudf.StructDtype))
        for dtype in dtypes
    ):
        # TODO: As list dtypes allow casting
        # to identical types, improve this logic of returning a
        # common dtype, for example:
        # ListDtype(int64) & ListDtype(int32) common
        # dtype could be ListDtype(int64).
        raise NotImplementedError(
            "Finding a common type for `ListDtype` or `StructDtype` is currently "
            "not supported"
        )

    # Corner case 1:
    # Resort to np.result_type to handle "M" and "m" types separately
    dt_dtypes = set(filter(lambda t: t.kind == "M", dtypes))
    if len(dt_dtypes) > 0:
        dtypes = dtypes - dt_dtypes
        dtypes.add(np.result_type(*dt_dtypes))

    td_dtypes = set(filter(lambda t: t.kind == "m", dtypes))
    if len(td_dtypes) > 0:
        dtypes = dtypes - td_dtypes
        dtypes.add(np.result_type(*td_dtypes))

    common_dtype = np.result_type(*dtypes)
    if common_dtype == np.dtype("float16"):
        return cudf.dtype("float32")
    return cudf.dtype(common_dtype)


def _dtype_pandas_compatible(dtype):
    """
    A utility function, that returns `str` instead of `object`
    dtype when pandas compatibility mode is enabled.
    """
    if cudf.get_option("mode.pandas_compatible") and dtype == cudf.dtype("O"):
        return "str"
    return dtype


def _maybe_convert_to_default_type(dtype: DtypeObj) -> DtypeObj:
    """Convert `dtype` to default if specified by user.

    If not specified, return as is.
    """
    if ib := cudf.get_option("default_integer_bitwidth"):
        if dtype.kind == "i":
            return cudf.dtype(f"i{ib//8}")
        elif dtype.kind == "u":
            return cudf.dtype(f"u{ib//8}")
    if (fb := cudf.get_option("default_float_bitwidth")) and dtype.kind == "f":
        return cudf.dtype(f"f{fb//8}")
    return dtype


def _get_base_dtype(dtype: pd.DatetimeTZDtype) -> np.dtype:
    # TODO: replace the use of this function with just `dtype.base`
    # when Pandas 2.1.0 is the minimum version we support:
    # https://github.com/pandas-dev/pandas/pull/52706
    if isinstance(dtype, pd.DatetimeTZDtype):
        return np.dtype(f"<M8[{dtype.unit}]")
    else:
        return dtype.base


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
