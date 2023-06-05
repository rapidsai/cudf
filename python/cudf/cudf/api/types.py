# Copyright (c) 2021-2023, NVIDIA CORPORATION.

"""Define common type operations."""

from __future__ import annotations

from collections import abc
from functools import wraps
from inspect import isclass
from typing import List, Union

import cupy as cp
import numpy as np
import pandas as pd
from pandas.api import types as pd_types

import cudf
from cudf.core.dtypes import (  # noqa: F401
    _BaseDtype,
    dtype,
    is_categorical_dtype,
    is_decimal32_dtype,
    is_decimal64_dtype,
    is_decimal128_dtype,
    is_decimal_dtype,
    is_interval_dtype,
    is_list_dtype,
    is_struct_dtype,
)


def is_numeric_dtype(obj):
    """Check whether the provided array or dtype is of a numeric dtype.

    Parameters
    ----------
    obj : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    bool
        Whether or not the array or dtype is of a numeric dtype.
    """
    if isclass(obj):
        if issubclass(obj, cudf.core.dtypes.DecimalDtype):
            return True
        if issubclass(obj, _BaseDtype):
            return False
    else:
        if isinstance(
            obj,
            (cudf.Decimal128Dtype, cudf.Decimal64Dtype, cudf.Decimal32Dtype),
        ) or isinstance(
            getattr(obj, "dtype", None),
            (cudf.Decimal128Dtype, cudf.Decimal64Dtype, cudf.Decimal32Dtype),
        ):
            return True
        if isinstance(obj, _BaseDtype) or isinstance(
            getattr(obj, "dtype", None), _BaseDtype
        ):
            return False
    if isinstance(obj, cudf.BaseIndex):
        return obj._is_numeric()
    return pd_types.is_numeric_dtype(obj)


# A version of numerical type check that does not include cudf decimals for
# places where we need to distinguish fixed and floating point numbers.
def _is_non_decimal_numeric_dtype(obj):
    if isinstance(obj, _BaseDtype) or isinstance(
        getattr(obj, "dtype", None), _BaseDtype
    ):
        return False
    try:
        return pd_types.is_numeric_dtype(obj)
    except TypeError:
        return False


def is_integer(obj):
    """Return True if given object is integer.

    Returns
    -------
    bool
    """
    if isinstance(obj, cudf.Scalar):
        return pd.api.types.is_integer_dtype(obj.dtype)
    return pd.api.types.is_integer(obj)


def is_string_dtype(obj):
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
        pd.api.types.is_string_dtype(obj)
        # Reject all cudf extension types.
        and not is_categorical_dtype(obj)
        and not is_decimal_dtype(obj)
        and not is_list_dtype(obj)
        and not is_struct_dtype(obj)
        and not is_interval_dtype(obj)
    )


def is_scalar(val):
    """Return True if given object is scalar.

    Parameters
    ----------
    val : object
        Possibly scalar object.

    Returns
    -------
    bool
        Return True if given object is scalar.
    """
    return isinstance(
        val,
        (
            cudf.Scalar,
            cudf._lib.scalar.DeviceScalar,
            cudf.core.tools.datetimes.DateOffset,
        ),
    ) or pd_types.is_scalar(val)


def _is_scalar_or_zero_d_array(val):
    """Return True if given object is scalar or a 0d array.

    This is an internal function primarily used by indexing applications that
    need to flatten dimensions that are indexed by 0d arrays.

    Parameters
    ----------
    val : object
        Possibly scalar object.

    Returns
    -------
    bool
        Return True if given object is scalar.
    """
    return (
        isinstance(val, (np.ndarray, cp.ndarray)) and val.ndim == 0
    ) or is_scalar(val)


# TODO: We should be able to reuse the pandas function for this, need to figure
# out why we can't.
def is_list_like(obj):
    """Return `True` if the given `obj` is list-like (list, tuple, Series...).

    Parameters
    ----------
    obj : object of any type which needs to be validated.

    Returns
    -------
    bool
        Return True if given object is list-like.
    """
    return isinstance(obj, (abc.Sequence, np.ndarray)) and not isinstance(
        obj, (str, bytes)
    )


# These methods are aliased directly into this namespace, but can be modified
# later if we determine that there is a need.


def _wrap_pandas_is_dtype_api(func):
    """Wrap a pandas dtype checking function to ignore cudf types."""

    @wraps(func)
    def wrapped_func(obj):
        if (
            (isclass(obj) and issubclass(obj, _BaseDtype))
            or isinstance(obj, _BaseDtype)
            or isinstance(getattr(obj, "dtype", None), _BaseDtype)
        ):
            return False
        return func(obj)

    return wrapped_func


def _union_categoricals(
    to_union: List[Union[cudf.Series, cudf.CategoricalIndex]],
    sort_categories: bool = False,
    ignore_order: bool = False,
):
    """Combine categorical data.

    This API is currently internal but should be exposed once full support for
    cudf.Categorical is ready.
    """
    # TODO(s) in the order specified :
    # 1. The return type needs to be changed
    #    to cudf.Categorical once it is implemented.
    # 2. Make this API public (i.e., to resemble
    #    pd.api.types.union_categoricals)

    if ignore_order:
        raise TypeError("ignore_order is not yet implemented")

    result_col = cudf.core.column.CategoricalColumn._concat(
        [obj._column for obj in to_union]
    )
    if sort_categories:
        sorted_categories = result_col.categories.sort_by_values(
            ascending=True
        )[0]
        result_col = result_col.reorder_categories(
            new_categories=sorted_categories
        )

    return cudf.Index(result_col)


def is_bool_dtype(arr_or_dtype):
    """
    Check whether the provided array or dtype is of a boolean dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a boolean dtype.

    Examples
    --------
    >>> from cudf.api.types import is_bool_dtype
    >>> import numpy as np
    >>> import cudf
    >>> is_bool_dtype(str)
    False
    >>> is_bool_dtype(int)
    False
    >>> is_bool_dtype(bool)
    True
    >>> is_bool_dtype(np.bool_)
    True
    >>> is_bool_dtype(np.array(['a', 'b']))
    False
    >>> is_bool_dtype(cudf.Series([1, 2]))
    False
    >>> is_bool_dtype(np.array([True, False]))
    True
    >>> is_bool_dtype(cudf.Series([True, False], dtype='category'))
    True
    """
    if isinstance(arr_or_dtype, cudf.BaseIndex):
        return arr_or_dtype._is_boolean()
    elif isinstance(arr_or_dtype, cudf.Series):
        if isinstance(arr_or_dtype.dtype, cudf.CategoricalDtype):
            return is_bool_dtype(arr_or_dtype=arr_or_dtype.dtype)
        else:
            return pd_types.is_bool_dtype(arr_or_dtype=arr_or_dtype.dtype)
    elif isinstance(arr_or_dtype, cudf.CategoricalDtype):
        return pd_types.is_bool_dtype(
            arr_or_dtype=arr_or_dtype.categories.dtype
        )
    else:
        return pd_types.is_bool_dtype(arr_or_dtype=arr_or_dtype)


def is_object_dtype(arr_or_dtype):
    """
    Check whether an array-like or dtype is of the object dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the object dtype.

    Examples
    --------
    >>> from cudf.api.types import is_object_dtype
    >>> import numpy as np
    >>> is_object_dtype(object)
    True
    >>> is_object_dtype(int)
    False
    >>> is_object_dtype(np.array([], dtype=object))
    True
    >>> is_object_dtype(np.array([], dtype=int))
    False
    >>> is_object_dtype([1, 2, 3])
    False
    """
    if isinstance(arr_or_dtype, cudf.BaseIndex):
        return arr_or_dtype._is_object()
    elif isinstance(arr_or_dtype, cudf.Series):
        return pd_types.is_object_dtype(arr_or_dtype=arr_or_dtype.dtype)
    else:
        return pd_types.is_object_dtype(arr_or_dtype=arr_or_dtype)


def is_float_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a float dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a float dtype.

    Examples
    --------
    >>> from cudf.api.types import is_float_dtype
    >>> import numpy as np
    >>> import cudf
    >>> is_float_dtype(str)
    False
    >>> is_float_dtype(int)
    False
    >>> is_float_dtype(float)
    True
    >>> is_float_dtype(np.array(['a', 'b']))
    False
    >>> is_float_dtype(cudf.Series([1, 2]))
    False
    >>> is_float_dtype(cudf.Index([1, 2.]))
    True
    """
    if isinstance(arr_or_dtype, cudf.BaseIndex):
        return arr_or_dtype._is_floating()
    return _wrap_pandas_is_dtype_api(pd_types.is_float_dtype)(arr_or_dtype)


def is_integer_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of an integer dtype.
    Unlike in `is_any_int_dtype`, timedelta64 instances will return False.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of an integer dtype and
        not an instance of timedelta64.

    Examples
    --------
    >>> from cudf.api.types import is_integer_dtype
    >>> import numpy as np
    >>> import cudf
    >>> is_integer_dtype(str)
    False
    >>> is_integer_dtype(int)
    True
    >>> is_integer_dtype(float)
    False
    >>> is_integer_dtype(np.uint64)
    True
    >>> is_integer_dtype('int8')
    True
    >>> is_integer_dtype('Int8')
    True
    >>> is_integer_dtype(np.datetime64)
    False
    >>> is_integer_dtype(np.timedelta64)
    False
    >>> is_integer_dtype(np.array(['a', 'b']))
    False
    >>> is_integer_dtype(cudf.Series([1, 2]))
    True
    >>> is_integer_dtype(np.array([], dtype=np.timedelta64))
    False
    >>> is_integer_dtype(cudf.Index([1, 2.]))  # float
    False
    """
    if isinstance(arr_or_dtype, cudf.BaseIndex):
        return arr_or_dtype._is_integer()
    return _wrap_pandas_is_dtype_api(pd_types.is_integer_dtype)(arr_or_dtype)


def is_any_real_numeric_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a real number dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a real number dtype.

    Examples
    --------
    >>> from cudf.api.types import is_any_real_numeric_dtype
    >>> import cudf
    >>> is_any_real_numeric_dtype(int)
    True
    >>> is_any_real_numeric_dtype(float)
    True
    >>> is_any_real_numeric_dtype(object)
    False
    >>> is_any_real_numeric_dtype(str)
    False
    >>> is_any_real_numeric_dtype(complex(1, 2))
    False
    >>> is_any_real_numeric_dtype(bool)
    False
    >>> is_any_real_numeric_dtype(cudf.Index([1, 2, 3]))
    True
    """
    return (
        is_numeric_dtype(arr_or_dtype)
        and not is_complex_dtype(arr_or_dtype)
        and not is_bool_dtype(arr_or_dtype)
    )


# TODO: The below alias is removed for now since improving cudf categorical
# support is ongoing and we don't want to introduce any ambiguities. The above
# method _union_categoricals will take its place once exposed.
# union_categoricals = pd_types.union_categoricals
infer_dtype = pd_types.infer_dtype
pandas_dtype = pd_types.pandas_dtype
is_complex_dtype = pd_types.is_complex_dtype
# TODO: Evaluate which of the datetime types need special handling for cudf.
is_datetime_dtype = _wrap_pandas_is_dtype_api(pd_types.is_datetime64_dtype)
is_datetime64_any_dtype = pd_types.is_datetime64_any_dtype
is_datetime64_dtype = pd_types.is_datetime64_dtype
is_datetime64_ns_dtype = pd_types.is_datetime64_ns_dtype
is_datetime64tz_dtype = pd_types.is_datetime64tz_dtype
is_extension_type = pd_types.is_extension_type
is_extension_array_dtype = pd_types.is_extension_array_dtype
is_int64_dtype = pd_types.is_int64_dtype
is_period_dtype = pd_types.is_period_dtype
is_signed_integer_dtype = pd_types.is_signed_integer_dtype
is_timedelta_dtype = _wrap_pandas_is_dtype_api(pd_types.is_timedelta64_dtype)
is_timedelta64_dtype = pd_types.is_timedelta64_dtype
is_timedelta64_ns_dtype = pd_types.is_timedelta64_ns_dtype
is_unsigned_integer_dtype = pd_types.is_unsigned_integer_dtype
is_sparse = pd_types.is_sparse
# is_list_like = pd_types.is_list_like
is_dict_like = pd_types.is_dict_like
is_file_like = pd_types.is_file_like
is_named_tuple = pd_types.is_named_tuple
is_iterator = pd_types.is_iterator
is_bool = pd_types.is_bool
is_categorical = pd_types.is_categorical
is_complex = pd_types.is_complex
is_float = pd_types.is_float
is_hashable = pd_types.is_hashable
is_interval = pd_types.is_interval
is_number = pd_types.is_number
is_re = pd_types.is_re
is_re_compilable = pd_types.is_re_compilable
is_dtype_equal = pd_types.is_dtype_equal


# Aliases of numpy dtype functionality.
issubdtype = np.issubdtype
