import numbers
from collections import namedtuple
from collections.abc import Sequence

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.types import pandas_dtype
from pandas.core.dtypes.dtypes import CategoricalDtype, CategoricalDtypeType

import cudf

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
OTHER_TYPES = {"bool", "category", "str"}
ALL_TYPES = NUMERIC_TYPES | DATETIME_TYPES | OTHER_TYPES


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


def is_numerical_dtype(obj):
    return not is_categorical_dtype(obj) and (
        np.issubdtype(obj, np.bool_)
        or np.issubdtype(obj, np.floating)
        or np.issubdtype(obj, np.signedinteger)
    )


def is_string_dtype(obj):
    return pd.api.types.is_string_dtype(obj) and not is_categorical_dtype(obj)


def is_datetime_dtype(obj):
    if obj is None:
        return False
    if not hasattr(obj, "str"):
        return False
    return "M8" in obj.str


def is_categorical_dtype(obj):
    """Infer whether a given pandas, numpy, or cuDF Column, Series, or dtype
    is a pandas CategoricalDtype.
    """
    if obj is None:
        return False
    if isinstance(obj, cudf.CategoricalDtype):
        return True
    if obj is cudf.CategoricalDtype:
        return True
    if isinstance(obj, np.dtype):
        return False
    if isinstance(obj, CategoricalDtype):
        return True
    if obj is CategoricalDtype:
        return True
    if obj is CategoricalDtypeType:
        return True
    if isinstance(obj, str) and obj == "category":
        return True
    if isinstance(
        obj,
        (
            CategoricalDtype,
            cudf.core.index.CategoricalIndex,
            cudf.core.column.CategoricalColumn,
            pd.Categorical,
            pd.CategoricalIndex,
        ),
    ):
        return True
    if isinstance(obj, np.ndarray):
        return False
    if isinstance(
        obj,
        (
            cudf.Index,
            cudf.Series,
            cudf.core.column.ColumnBase,
            pd.Index,
            pd.Series,
        ),
    ):
        return is_categorical_dtype(obj.dtype)
    if hasattr(obj, "type"):
        if obj.type is CategoricalDtypeType:
            return True
    return pandas_dtype(obj).type is CategoricalDtypeType


def cudf_dtype_from_pydata_dtype(dtype):
    """ Given a numpy or pandas dtype, converts it into the equivalent cuDF
        Python dtype.
    """
    from pandas.core.dtypes.common import infer_dtype_from_object

    if is_categorical_dtype(dtype):
        return cudf.core.dtypes.CategoricalDtype
    elif np.issubdtype(dtype, np.datetime64):
        dtype = np.datetime64

    return infer_dtype_from_object(dtype)


def is_scalar(val):
    return (
        val is None
        or isinstance(val, str)
        or isinstance(val, numbers.Number)
        or np.isscalar(val)
        or (isinstance(val, (np.ndarray, cp.ndarray)) and val.ndim == 0)
        or isinstance(val, pd.Timestamp)
        or (isinstance(val, pd.Categorical) and len(val) == 1)
    )


def to_cudf_compatible_scalar(val, dtype=None):
    """
    Converts the value `val` to a numpy/Pandas scalar,
    optionally casting to `dtype`.

    If `val` is None, returns None.
    """
    if val is None:
        return val

    if not is_scalar(val):
        raise ValueError(
            f"Cannot convert value of type {type(val).__name__} "
            " to cudf scalar"
        )

    if isinstance(val, (np.ndarray, cp.ndarray)) and val.ndim == 0:
        val = val.item()

    if ((dtype is None) and isinstance(val, str)) or is_string_dtype(dtype):
        dtype = "str"

    val = pd.api.types.pandas_dtype(type(val)).type(val)

    if dtype is not None:
        val = val.astype(dtype)

    if val.dtype.type is np.datetime64:
        time_unit, _ = np.datetime_data(val.dtype)
        if time_unit in ("D", "W", "M", "Y"):
            val = val.astype("datetime64[s]")

    return val


def is_list_like(obj):
    """
    This function checks if the given `obj`
    is a list-like (list, tuple, Series...)
    type or not.

    Parameters
    ----------
    obj : object of any type which needs to be validated.

    Returns
    -------
    Boolean: True or False depending on whether the
    input `obj` is like-like or not.
    """
    return isinstance(obj, (Sequence,),) and not isinstance(obj, (str, bytes))


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


def min_column_type(x, expected_type):
    """
    Return the smallest dtype which can represent all
    elements of the `NumericalColumn` `x`
    If the column is not a subtype of `np.signedinteger` or `np.floating`
    returns the same dtype as the dtype of `x` without modification
    """
    from cudf.core.column import NumericalColumn

    if not isinstance(x, NumericalColumn):
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


def check_cast_unsupported_dtype(dtype):
    from cudf._lib.types import np_to_cudf_types

    if is_categorical_dtype(dtype):
        return dtype

    if isinstance(dtype, pd.core.arrays.numpy_.PandasDtype):
        dtype = dtype.numpy_dtype
    else:
        dtype = np.dtype(dtype)

    if dtype in np_to_cudf_types:
        return dtype

    if dtype == np.dtype("float16"):
        return np.dtype("float32")

    raise NotImplementedError(
        "Cannot cast {0} dtype, as it is not supported by CuDF.".format(dtype)
    )
