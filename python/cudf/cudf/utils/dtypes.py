# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api import types as pd_types  # noqa: TID251

import pylibcudf as plc

import cudf

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cudf._typing import DtypeObj
    from cudf.core.dtypes import DecimalDtype

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
pandas_dtypes_to_np_dtypes[pd.StringDtype("pyarrow")] = np.dtype("object")

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


def _find_common_type_decimal(dtypes: Iterable[DecimalDtype]) -> DecimalDtype:
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


def cudf_dtype_to_pa_type(dtype: DtypeObj) -> pa.DataType:
    """Given a cudf pandas dtype, converts it into the equivalent cuDF
    Python dtype.
    """
    dtype = getattr(dtype, "numpy_dtype", dtype)
    if isinstance(dtype, cudf.CategoricalDtype):
        raise NotImplementedError(
            "No conversion from Categorical to pyarrow type"
        )
    elif isinstance(
        dtype,
        (cudf.StructDtype, cudf.ListDtype, cudf.core.dtypes.DecimalDtype),
    ):
        return dtype.to_arrow()
    elif isinstance(dtype, pd.DatetimeTZDtype):
        return pa.timestamp(dtype.unit, str(dtype.tz))
    elif dtype == CUDF_STRING_DTYPE or isinstance(dtype, pd.StringDtype):
        return pa.string()
    else:
        return pa.from_numpy_dtype(dtype)


def cudf_dtype_from_pa_type(typ: pa.DataType) -> DtypeObj:
    """Given a cuDF pyarrow dtype, converts it into the equivalent
    cudf pandas dtype.
    """
    if pa.types.is_list(typ):
        return cudf.core.dtypes.ListDtype.from_arrow(typ)
    elif pa.types.is_struct(typ):
        return cudf.core.dtypes.StructDtype.from_arrow(typ)
    elif pa.types.is_decimal(typ):
        return cudf.core.dtypes.Decimal128Dtype.from_arrow(typ)
    elif pa.types.is_large_string(typ) or pa.types.is_string(typ):
        return CUDF_STRING_DTYPE
    else:
        return cudf.api.types.pandas_dtype(typ.to_pandas_dtype())


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
        dtype = np.dtype(int_dtype)
        if (dtype.itemsize * 8) >= min_size:
            if np.iinfo(int_dtype).min <= x <= np.iinfo(int_dtype).max:
                return dtype
    # resort to using `int64` and let numpy raise appropriate exception:
    return np.int64(x).dtype


def min_unsigned_type(x: int, min_size: int = 8) -> np.dtype:
    """
    Return the smallest *unsigned* integer dtype
    that can represent the integer ``x``
    """
    for int_dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        dtype = np.dtype(int_dtype)
        if (dtype.itemsize * 8) >= min_size:
            if 0 <= x <= np.iinfo(int_dtype).max:
                return dtype
    # resort to using `uint64` and let numpy raise appropriate exception:
    return np.uint64(x).dtype


def is_mixed_with_object_dtype(lhs, rhs):
    if isinstance(lhs.dtype, cudf.CategoricalDtype):
        return is_mixed_with_object_dtype(lhs.dtype.categories, rhs)
    elif isinstance(rhs.dtype, cudf.CategoricalDtype):
        return is_mixed_with_object_dtype(lhs, rhs.dtype.categories)

    return (lhs.dtype == "object" and rhs.dtype != "object") or (
        rhs.dtype == "object" and lhs.dtype != "object"
    )


def _get_nan_for_dtype(dtype: DtypeObj) -> DtypeObj:
    if dtype.kind in "mM":
        time_unit, _ = np.datetime_data(dtype)
        return dtype.type("nat", time_unit)
    elif dtype.kind == "f":
        if is_pandas_nullable_extension_dtype(dtype):
            return dtype.na_value
        return dtype.type("nan")
    else:
        if (
            is_pandas_nullable_extension_dtype(dtype)
            and getattr(dtype, "kind", "c") in "biu"
        ):
            return dtype.na_value
        return np.float64("nan")


def find_common_type(dtypes: Iterable[DtypeObj]) -> DtypeObj | None:
    """
    Wrapper over np.result_type to handle cudf specific types.

    Parameters
    ----------
    dtypes : iterable
        sequence of dtypes to find common types

    Returns
    -------
    dtype : np.dtype or None
        None if input is empty
        DtypeObj otherwise
    """
    if len(dtypes) == 0:  # type: ignore[arg-type]
        return None

    # Early exit for categoricals since they're not hashable and therefore
    # can't be put in a set.
    if any(isinstance(dtype, cudf.CategoricalDtype) for dtype in dtypes):
        if all(
            (
                isinstance(dtype, cudf.CategoricalDtype)
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
                raise NotImplementedError(
                    "Only unordered categories of the same underlying type "
                    "may be currently coerced to a common type."
                )
        else:
            # TODO: Should this be an error case (mixing categorical with other
            # dtypes) or should this return object? Unclear if we have enough
            # information to decide right now, may have to come back to this as
            # usage of find_common_type increases.
            return CUDF_STRING_DTYPE

    # Aggregate same types
    dtypes = set(dtypes)
    if len(dtypes) == 1:
        return dtypes.pop()

    if any(
        isinstance(dtype, cudf.core.dtypes.DecimalDtype) for dtype in dtypes
    ):
        if all(
            is_dtype_obj_numeric(dtype, include_decimal=True)
            for dtype in dtypes
        ):
            return _find_common_type_decimal(
                [
                    dtype
                    for dtype in dtypes
                    if isinstance(dtype, cudf.core.dtypes.DecimalDtype)
                ]
            )
        else:
            return CUDF_STRING_DTYPE
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

    common_dtype = np.result_type(*dtypes)  # noqa: TID251
    if common_dtype == np.dtype(np.float16):
        return np.dtype(np.float32)
    return common_dtype


def _dtype_pandas_compatible(dtype):
    """
    A utility function, that returns `str` instead of `object`
    dtype when pandas compatibility mode is enabled.
    """
    if (
        cudf.get_option("mode.pandas_compatible")
        and dtype == CUDF_STRING_DTYPE
    ):
        return "str"
    return dtype


def _maybe_convert_to_default_type(dtype: DtypeObj) -> DtypeObj:
    """Convert `dtype` to default if specified by user.

    If not specified, return as is.
    """
    if ib := cudf.get_option("default_integer_bitwidth"):
        if dtype.kind == "i":
            return np.dtype(f"i{ib // 8}")
        elif dtype.kind == "u":
            return np.dtype(f"u{ib // 8}")
    if (fb := cudf.get_option("default_float_bitwidth")) and dtype.kind == "f":
        return np.dtype(f"f{fb // 8}")
    return dtype


def _get_base_dtype(dtype: pd.DatetimeTZDtype) -> np.dtype:
    # TODO: replace the use of this function with just `dtype.base`
    # when Pandas 2.1.0 is the minimum version we support:
    # https://github.com/pandas-dev/pandas/pull/52706
    if isinstance(dtype, pd.DatetimeTZDtype):
        return np.dtype(f"<M8[{dtype.unit}]")
    else:
        return dtype.base


def is_dtype_obj_numeric(
    dtype: DtypeObj, include_decimal: bool = True
) -> bool:
    """Like is_numeric_dtype but does not introspect argument."""
    is_non_decimal = dtype.kind in set("iufb")
    if include_decimal:
        return is_non_decimal or isinstance(
            dtype,
            (cudf.Decimal32Dtype, cudf.Decimal64Dtype, cudf.Decimal128Dtype),
        )
    else:
        return is_non_decimal


pa_decimal32type = getattr(pa, "Decimal32Type", None)

pa_decimal64type = getattr(pa, "Decimal64Type", None)


def pyarrow_dtype_to_cudf_dtype(dtype: pa.DataType) -> DtypeObj:
    """Given a pyarrow dtype, converts it into the equivalent cudf pandas
    dtype.
    """

    pyarrow_dtype = dtype.pyarrow_dtype
    if isinstance(pyarrow_dtype, pa.Decimal128Type):
        return cudf.Decimal128Dtype.from_arrow(pyarrow_dtype)
    elif pa_decimal64type is not None and isinstance(
        pyarrow_dtype, pa_decimal64type
    ):
        return cudf.Decimal64Dtype.from_arrow(pyarrow_dtype)
    elif pa_decimal32type is not None and isinstance(
        pyarrow_dtype, pa_decimal32type
    ):
        return cudf.Decimal32Dtype.from_arrow(pyarrow_dtype)
    elif isinstance(pyarrow_dtype, pa.ListType):
        return cudf.ListDtype.from_arrow(pyarrow_dtype)
    elif isinstance(pyarrow_dtype, pa.StructType):
        return cudf.StructDtype.from_arrow(pyarrow_dtype)
    elif str(pyarrow_dtype) == "large_string":
        return CUDF_STRING_DTYPE
    elif pyarrow_dtype is pa.date32():
        raise TypeError("Unsupported type")
    elif isinstance(pyarrow_dtype, pa.DataType):
        return pyarrow_dtype.to_pandas_dtype()
    else:
        raise TypeError(f"Unsupported Arrow type: {pyarrow_dtype}")


def is_pandas_nullable_numpy_dtype(dtype_to_check) -> bool:
    return isinstance(
        dtype_to_check,
        (
            pd.UInt8Dtype,
            pd.UInt16Dtype,
            pd.UInt32Dtype,
            pd.UInt64Dtype,
            pd.Int8Dtype,
            pd.Int16Dtype,
            pd.Int32Dtype,
            pd.Int64Dtype,
            pd.Float32Dtype,
            pd.Float64Dtype,
            pd.BooleanDtype,
            pd.StringDtype,
        ),
    )


def is_pandas_nullable_extension_dtype(dtype_to_check) -> bool:
    if is_pandas_nullable_numpy_dtype(dtype_to_check) or isinstance(
        dtype_to_check, pd.ArrowDtype
    ):
        return True
    elif isinstance(dtype_to_check, pd.CategoricalDtype):
        if dtype_to_check.categories is None:
            return False
        return is_pandas_nullable_extension_dtype(
            dtype_to_check.categories.dtype
        )
    elif isinstance(dtype_to_check, pd.IntervalDtype):
        return is_pandas_nullable_extension_dtype(dtype_to_check.subtype)
    return False


def dtype_to_pylibcudf_type(dtype) -> plc.DataType:
    if isinstance(dtype, pd.ArrowDtype):
        dtype = pyarrow_dtype_to_cudf_dtype(dtype)
    if isinstance(dtype, cudf.ListDtype):
        return plc.DataType(plc.TypeId.LIST)
    elif isinstance(dtype, cudf.StructDtype):
        return plc.DataType(plc.TypeId.STRUCT)
    elif isinstance(dtype, cudf.Decimal128Dtype):
        tid = plc.TypeId.DECIMAL128
        return plc.DataType(tid, -dtype.scale)
    elif isinstance(dtype, cudf.Decimal64Dtype):
        tid = plc.TypeId.DECIMAL64
        return plc.DataType(tid, -dtype.scale)
    elif isinstance(dtype, cudf.Decimal32Dtype):
        tid = plc.TypeId.DECIMAL32
        return plc.DataType(tid, -dtype.scale)
    # libcudf types don't support timezones so convert to the base type
    elif isinstance(dtype, pd.DatetimeTZDtype):
        dtype = _get_base_dtype(dtype)
    elif isinstance(dtype, pd.StringDtype):
        dtype = CUDF_STRING_DTYPE
    else:
        dtype = pandas_dtypes_to_np_dtypes.get(dtype, dtype)
        try:
            dtype = np.dtype(dtype)
        except TypeError:
            dtype = cudf.dtype(dtype)
    return plc.DataType(SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES[dtype])


def dtype_to_pandas_arrowdtype(dtype) -> pd.ArrowDtype:
    if isinstance(dtype, pd.ArrowDtype):
        return dtype
    if isinstance(
        dtype,
        (cudf.ListDtype, cudf.StructDtype, cudf.core.dtypes.DecimalDtype),
    ):
        return pd.ArrowDtype(dtype.to_arrow())
    # libcudf types don't support timezones so convert to the base type
    elif isinstance(dtype, pd.DatetimeTZDtype):
        dtype = _get_base_dtype(dtype)
    else:
        dtype = pandas_dtypes_to_np_dtypes.get(dtype, dtype)
        try:
            dtype = np.dtype(dtype)
        except TypeError:
            dtype = cudf.dtype(dtype)
    if dtype is CUDF_STRING_DTYPE:
        dtype = np.dtype("str")
    return pd.ArrowDtype(pa.from_numpy_dtype(dtype))


def dtype_to_pandas_nullable_extension_type(dtype) -> DtypeObj:
    if isinstance(dtype, pd.ArrowDtype):
        return dtype_to_pandas_nullable_extension_type(
            pyarrow_dtype_to_cudf_dtype(dtype)
        )
    else:
        return np_dtypes_to_pandas_dtypes.get(dtype, dtype)


def get_dtype_of_same_kind(source_dtype: DtypeObj, target_dtype: DtypeObj):
    """
    Given a dtype, return a dtype of the same kind.
    If no such dtype exists, return the default dtype.
    """
    if isinstance(source_dtype, pd.ArrowDtype):
        return dtype_to_pandas_arrowdtype(target_dtype)
    elif is_pandas_nullable_extension_dtype(source_dtype):
        return dtype_to_pandas_nullable_extension_type(target_dtype)
    else:
        return target_dtype


def get_dtype_of_same_type(lhs_dtype: DtypeObj, rhs_dtype: DtypeObj):
    """
    Given two dtypes, checks if `lhs_dtype` translates to same libcudf
    type as `rhs_dtype`, if yes, returns `lhs_dtype`.
    Else, returns `rhs_dtype` in `lhs_dtype`'s kind.
    """
    if dtype_to_pylibcudf_type(lhs_dtype) == dtype_to_pylibcudf_type(
        rhs_dtype
    ):
        return lhs_dtype
    else:
        return get_dtype_of_same_kind(lhs_dtype, rhs_dtype)


def dtype_from_pylibcudf_column(col: plc.Column) -> DtypeObj:
    type_ = col.type()
    tid = type_.id()

    if tid == plc.TypeId.LIST:
        child = col.list_view().child()
        return cudf.ListDtype(dtype_from_pylibcudf_column(child))
    elif tid == plc.TypeId.STRUCT:
        fields = {
            str(i): dtype_from_pylibcudf_column(col.child(i))
            for i in range(col.num_children())
        }
        return cudf.StructDtype(fields)
    elif tid == plc.TypeId.DECIMAL64:
        return cudf.Decimal64Dtype(
            precision=cudf.Decimal64Dtype.MAX_PRECISION, scale=-type_.scale()
        )
    elif tid == plc.TypeId.DECIMAL32:
        return cudf.Decimal32Dtype(
            precision=cudf.Decimal32Dtype.MAX_PRECISION, scale=-type_.scale()
        )
    elif tid == plc.TypeId.DECIMAL128:
        return cudf.Decimal128Dtype(
            precision=cudf.Decimal128Dtype.MAX_PRECISION, scale=-type_.scale()
        )
    else:
        return PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[tid]


def is_dtype_obj_categorical(obj):
    if obj is None:
        return False

    if isinstance(
        obj,
        (
            pd.CategoricalDtype,
            cudf.CategoricalDtype,
        ),
    ):
        return True

    if any(
        obj is t
        for t in (
            cudf.CategoricalDtype,
            pd.CategoricalDtype,
            pd.CategoricalDtype.type,
        )
    ):
        return True
    if isinstance(obj, str) and obj == "category":
        return True

    # TODO: A lot of the above checks are probably redundant and should be
    # farmed out to this function here instead.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd_types.is_categorical_dtype(obj)


def is_dtype_obj_string(obj):
    """Check whether the provided array or dtype is of the string dtype.

    Parameters
    ----------
    obj : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    bool
        Whether or not the array or dtype is of the string dtype.
    """
    return (
        obj is CUDF_STRING_DTYPE
        or (isinstance(obj, pd.StringDtype))
        or (
            isinstance(obj, pd.ArrowDtype)
            and (
                pa.types.is_string(obj.pyarrow_dtype)
                or pa.types.is_large_string(obj.pyarrow_dtype)
            )
        )
    )


def is_dtype_obj_list(obj):
    """Check whether an array-like or dtype is of the list dtype.

    Parameters
    ----------
    obj : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    bool
        Whether or not the array-like or dtype is of the list dtype.
    """
    return (
        type(obj) is cudf.ListDtype
        or obj is cudf.ListDtype
        or (isinstance(obj, str) and obj == cudf.ListDtype.name)
        or (
            isinstance(obj, pd.ArrowDtype)
            and pa.types.is_list(obj.pyarrow_dtype)
        )
    )


def is_dtype_obj_struct(obj):
    """Check whether an array-like or dtype is of the struct dtype.

    Parameters
    ----------
    obj : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    bool
        Whether or not the array-like or dtype is of the struct dtype.
    """
    # TODO: This behavior is currently inconsistent for interval types. the
    # actual class IntervalDtype will return False, but instances (e.g.
    # IntervalDtype(int)) will return True. For now this is not being changed
    # since the interval dtype is being modified as part of the array refactor,
    # but this behavior should be made consistent afterwards.
    return (
        isinstance(obj, cudf.StructDtype)
        or obj is cudf.StructDtype
        or (isinstance(obj, str) and obj == cudf.StructDtype.name)
        or (
            isinstance(obj, pd.ArrowDtype)
            and pa.types.is_struct(obj.pyarrow_dtype)
        )
    )


def is_dtype_obj_interval(obj):
    return (
        isinstance(
            obj,
            (
                cudf.IntervalDtype,
                pd.IntervalDtype,
            ),
        )
        or obj is cudf.IntervalDtype
        or (isinstance(obj, str) and obj == cudf.IntervalDtype.name)
        or (
            isinstance(obj, pd.ArrowDtype)
            and pa.types.is_interval(obj.pyarrow_dtype)
        )
    )


def is_dtype_obj_decimal(obj):
    """Check whether an array-like or dtype is of the decimal dtype.

    Parameters
    ----------
    obj : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    bool
        Whether or not the array-like or dtype is of the decimal dtype.
    """
    return (
        is_dtype_obj_decimal32(obj)
        or is_dtype_obj_decimal64(obj)
        or is_dtype_obj_decimal128(obj)
    )


def is_dtype_obj_decimal32(obj):
    return (
        type(obj) is cudf.Decimal32Dtype
        or obj is cudf.Decimal32Dtype
        or (isinstance(obj, str) and obj == cudf.Decimal32Dtype.name)
    )


def is_dtype_obj_decimal64(obj):
    return (
        type(obj) is cudf.Decimal64Dtype
        or obj is cudf.Decimal64Dtype
        or (isinstance(obj, str) and obj == cudf.Decimal64Dtype.name)
    )


def is_dtype_obj_decimal128(obj):
    return (
        type(obj) is cudf.Decimal128Dtype
        or obj is cudf.Decimal128Dtype
        or (isinstance(obj, str) and obj == cudf.Decimal128Dtype.name)
        or (
            isinstance(obj, pd.ArrowDtype)
            and pa.types.is_decimal128(obj.pyarrow_dtype)
        )
    )


SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES = {
    np.dtype("int8"): plc.types.TypeId.INT8,
    np.dtype("int16"): plc.types.TypeId.INT16,
    np.dtype("int32"): plc.types.TypeId.INT32,
    np.dtype("int64"): plc.types.TypeId.INT64,
    np.dtype("uint8"): plc.types.TypeId.UINT8,
    np.dtype("uint16"): plc.types.TypeId.UINT16,
    np.dtype("uint32"): plc.types.TypeId.UINT32,
    np.dtype("uint64"): plc.types.TypeId.UINT64,
    np.dtype("float32"): plc.types.TypeId.FLOAT32,
    np.dtype("float64"): plc.types.TypeId.FLOAT64,
    np.dtype("datetime64[s]"): plc.types.TypeId.TIMESTAMP_SECONDS,
    np.dtype("datetime64[ms]"): plc.types.TypeId.TIMESTAMP_MILLISECONDS,
    np.dtype("datetime64[us]"): plc.types.TypeId.TIMESTAMP_MICROSECONDS,
    np.dtype("datetime64[ns]"): plc.types.TypeId.TIMESTAMP_NANOSECONDS,
    np.dtype("object"): plc.types.TypeId.STRING,
    np.dtype("bool"): plc.types.TypeId.BOOL8,
    np.dtype("timedelta64[s]"): plc.types.TypeId.DURATION_SECONDS,
    np.dtype("timedelta64[ms]"): plc.types.TypeId.DURATION_MILLISECONDS,
    np.dtype("timedelta64[us]"): plc.types.TypeId.DURATION_MICROSECONDS,
    np.dtype("timedelta64[ns]"): plc.types.TypeId.DURATION_NANOSECONDS,
}
PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES = {
    plc_type: np_type
    for np_type, plc_type in SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES.items()
}
# There's no equivalent to EMPTY in cudf.  We translate EMPTY
# columns from libcudf to ``int8`` columns of all nulls in Python.
# ``int8`` is chosen because it uses the least amount of memory.
PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[plc.types.TypeId.EMPTY] = np.dtype("int8")
PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[plc.types.TypeId.STRUCT] = np.dtype(
    "object"
)
PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[plc.types.TypeId.LIST] = np.dtype("object")


SIZE_TYPE_DTYPE = PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[plc.types.SIZE_TYPE_ID]
CUDF_STRING_DTYPE = PYLIBCUDF_TO_SUPPORTED_NUMPY_TYPES[plc.types.TypeId.STRING]
