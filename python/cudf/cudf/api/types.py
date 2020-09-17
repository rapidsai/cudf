# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype, CategoricalDtypeType

import cudf


def is_bool_dtype(obj):
    """
    Check whether the provided array or dtype is of a boolean dtype.
    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.
    Returns
    -------
    boolean
        Whether or not the array or dtype is of a boolean dtype.
    Notes
    -----
    Accepts cuDF, Pandas, or NumPy dtypes and arrays.

    Examples
    --------
    >>> is_bool_dtype(cudf.BooleanDtype())
    True
    >>> is_bool_dtype(cudf.Series([True, False, None]))
    True
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
    >>> is_bool_dtype(pd.Series([1, 2]))
    False
    >>> is_bool_dtype(np.array([True, False]))
    True
    >>> is_bool_dtype(pd.Categorical([True, False]))
    True
    >>> is_bool_dtype(pd.arrays.SparseArray([True, False]))
    True
    """
    if hasattr(obj, 'dtype'):
        obj = obj.dtype
    if hasattr(obj, "dtype"):
        obj = obj.dtype
    return isinstance(obj, cudf.BooleanDtype) or (not isinstance(obj, cudf.Generic) and pd.api.types.is_bool_dtype(
        obj
    ))


def is_datetime64_dtype(obj):
    """
    Check whether the provided array or dtype is of the datetime64 dtype.
    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.
    Returns
    -------
    boolean
        Whether or not the array or dtype is of the datetime64 dtype.
    Notes
    --------
        Accepts cuDF, Pandas, or NumPy dtypes and arrays.

    Examples
    --------
    >>> is_datetime64_dtype(cudf.Datetime64NSDtype())
    True
    >>> is_datetime64_dtype(cudf.Series([1, 2, 3], dtype='datetime64[ms]'))
    True
    >>> is_datetime64_dtype(object)
    False
    >>> is_datetime64_dtype(np.datetime64)
    True
    >>> is_datetime64_dtype(np.array([], dtype=int))
    False
    >>> is_datetime64_dtype(np.array([], dtype=np.datetime64))
    True
    >>> is_datetime64_dtype([1, 2, 3])
    False
    """
    if hasattr(obj, 'dtype'):
        obj = obj.dtype
    return isinstance(obj, cudf.Datetime) or (not isinstance(obj, cudf.Generic) and pd.api.types.is_datetime64_dtype(
        obj
    ))


def is_timedelta64_dtype(obj):
    """
    Check whether an array or dtype is of the timedelta64 dtype.
    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.
    Returns
    -------
    boolean
        Whether or not the array or dtype is of the timedelta64 dtype.
    Examples
    --------
    >>> is_timedelta64_dtype(cudf.Timedelta64NSDtype())
    True
    >>> is_timedelta64_dtype(cudf.Series([1,2,3], dtype='timedelta64[ns]'))
    True
    >>> is_timedelta64_dtype(object)
    False
    >>> is_timedelta64_dtype(np.timedelta64)
    True
    >>> is_timedelta64_dtype([1, 2, 3])
    False
    >>> is_timedelta64_dtype(pd.Series([], dtype="timedelta64[ns]"))
    True
    >>> is_timedelta64_dtype('0 days')
    False
    """
    if hasattr(obj, 'dtype'):
        obj = obj.dtype
    return isinstance(
        obj, cudf.Timedelta
    ) or (not isinstance(obj, cudf.Generic) and pd.api.types.is_timedelta64_dtype(obj))


def is_string_dtype(obj):
    """
    Check whether the provided array or dtype is of the string dtype.
    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.
    Returns
    -------
    boolean
        Whether or not the array or dtype is of the string dtype.
    Examples
    --------
    >>> is_string_dtype(cudf.StringDtype())
    True
    >>> is_string_dtype(cudf.Series(['a','b','c']))
    True
    >>> is_string_dtype(str)
    True
    >>> is_string_dtype(object)
    True
    >>> is_string_dtype(int)
    False
    >>>
    >>> is_string_dtype(np.array(['a', 'b']))
    True
    >>> is_string_dtype(pd.Series([1, 2]))
    False
    """
    if hasattr(obj, 'dtype'):
        obj = obj.dtype
    return isinstance(obj, cudf.StringDtype) or (not isinstance(obj, cudf.Generic) and (
        pd.api.types.is_string_dtype(obj) and not is_categorical_dtype(obj)
    ))


def is_integer_dtype(obj):
    """
    Check whether the provided array or dtype is of an integer dtype.
    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.
    Returns
    -------
    boolean
        Whether or not the array or dtype is of an integer dtype and
    Examples
    --------
    >>> is_integer_dtype(cudf.Int64Dtype())
    True
    >>> is_integer_dtype(cudf.Series([1,2,3], dtype='int64'))
    True
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
    >>> is_integer_dtype(pd.Int8Dtype)
    True
    >>> is_integer_dtype(np.datetime64)
    False
    >>> is_integer_dtype(np.timedelta64)
    False
    >>> is_integer_dtype(np.array(['a', 'b']))
    False
    >>> is_integer_dtype(pd.Series([1, 2]))
    True
    >>> is_integer_dtype(np.array([], dtype=np.timedelta64))
    False
    >>> is_integer_dtype(pd.Index([1, 2.]))  # float
    False
    """
    if hasattr(obj, 'dtype'):
        obj = obj.dtype
    return isinstance(obj, cudf.Integer) or (not isinstance(obj, cudf.Generic) and pd.api.types.is_integer_dtype(obj))

def is_numeric_dtype(obj):
    """
    Check whether the provided array or dtype is of a numeric dtype.
    Parameters
    ----------
    arr_or_dtype : array-like
        The array or dtype to check.
    Returns
    -------
    boolean
        Whether or not the array or dtype is of a numeric dtype.
    Examples
    --------
    >>> is_numeric_dtype(cudf.Float32Dtype())
    True
    >>> is_numeric_dtype(cudf.Series([1.0, 2.0, 3.0]))
    True
    >>> is_numeric_dtype(str)
    False
    >>> is_numeric_dtype(int)
    True
    >>> is_numeric_dtype(float)
    True
    >>> is_numeric_dtype(np.uint64)
    True
    >>> is_numeric_dtype(np.datetime64)
    False
    >>> is_numeric_dtype(np.timedelta64)
    False
    >>> is_numeric_dtype(np.array(['a', 'b']))
    False
    >>> is_numeric_dtype(pd.Series([1, 2]))
    True
    >>> is_numeric_dtype(pd.Index([1, 2.]))
    True
    >>> is_numeric_dtype(np.array([], dtype=np.timedelta64))
    False
    """
    if hasattr(obj, 'dtype'):
        obj = obj.dtype
    if isinstance(obj, cudf.Generic):
        return isinstance(obj, (cudf.Number, cudf.BooleanDtype))
    if is_categorical_dtype(obj):
        return False
    if is_list_dtype(obj):
        return False
    return (
        np.issubdtype(obj, np.bool_)
        or np.issubdtype(obj, np.floating)
        or np.issubdtype(obj, np.signedinteger)
    )


def is_categorical_dtype(obj):
    """Infer whether a given pandas, numpy, or cuDF Column, Series, or dtype
    is a pandas CategoricalDtype.
    """
    if isinstance(obj, cudf.Generic) and not isinstance(
        obj, cudf.CategoricalDtype
    ):
        return False
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
    return pd.api.types.is_categorical_dtype(obj)


def is_list_dtype(obj):
    return (
        isinstance(obj, cudf.core.dtypes.ListDtype)
        or obj is cudf.core.dtypes.ListDtype
        or type(obj) is cudf.core.column.ListColumn
        or obj is cudf.core.column.ListColumn
        or (isinstance(obj, str) and obj == cudf.core.dtypes.ListDtype.name)
        or (hasattr(obj, "dtype") and is_list_dtype(obj.dtype))
    )


def find_common_type(array_types, scalar_types):
    """
    Determine common type following numpy coercion rules.
    Similar to numpy.find_common_type, but accepts both 
    numpy and cuDF datatypes.

    Parameters
    ----------
    array_types : sequence
        A list of dtypes or dtype convertible objects representing arrays.
    scalar_types : sequence
        A list of dtypes or dtype convertible objects representing scalars.
    Returns
    -------
    datatype : cuDF dtype
        The common data type, which is the maximum of `array_types` ignoring
        `scalar_types`, unless the maximum of `scalar_types` is of a
        different kind (`dtype.kind`). 
    See Also
    --------
    numpy.find_common_type
    Notes
    --------
    Accepts numpy dtypes, cuDF dtypes, or a mix of both

    """
    array_types = [
        d.numpy_dtype if isinstance(d, cudf.Generic) else d for d in array_types
    ]
    scalar_types = [
        d.numpy_dtype if isinstance(d, cudf.Generic) else d for d in scalar_types
    ]

    return cudf.dtype(np.find_common_type(array_types, scalar_types))


def can_cast(from_, to, casting='safe'):
    """
    Returns True if cast between data types can occur according to the casting rule.
    If from is a scalar or array scalar, also returns True if the scalar value
    can be cast without overflow or truncation to an integer.

    Parameters
    ----------
    from_ : dtype, dtype specifier, scalar, or array
        Data type, scalar, or array to cast from.
    to : dtype or dtype specifier
        Data type to cast to.
    casting : {‘no’, ‘equiv’, ‘safe’, ‘same_kind’, ‘unsafe’}, optional
        Controls what kind of data casting may occur.
        - ‘no’ means the data types should not be cast at all.
        - ‘equiv’ means only byte-order changes are allowed.
        - ‘safe’ means only casts which can preserve values are allowed.
        - ‘same_kind’ means only safe casts or casts within a kind, 
        like float64 to float32, are allowed
        - ‘unsafe’ means any data conversions may be done.

    Notes
    --------
    Accepts numpy dtypes, cuDF dtypes, or a mix of both
    """
    if isinstance(from_, cudf.Generic):
        from_ = from_.numpy_dtype
    elif isinstance(from_, cudf.Scalar):
        from_ = from_.value
    if isinstance(to, cudf.Generic):
        to = to.numpy_dtype

    return np.can_cast(from_, to, casting=casting)


def result_type(*arrays_and_dtypes):
    """
    Returns the type that results from applying the NumPy type promotion rules to the arguments.
    See numpy.result_type for details. 
    
    See Also
    --------
    numpy.result_type

    Returns
    -------
    datatype : cuDF dtype

    Notes
    --------
    Accepts numpy dtypes, cuDF dtypes, or a mix of both

    """
    arrays_and_dtypes = (
        d.numpy_dtype if isinstance(d, cudf.Generic) else d
        for d in arrays_and_dtypes
    )
    return cudf.dtype(np.result_type(*arrays_and_dtypes))

def isnan(x):
    """
    Returns true if an input scalar is equal to NaN.

    Parameters
    -------
    x : cuDF or NumPy scalar

    See Also
    -------
    numpy.isnan

    Notes
    --------
    Accepts numpy dtypes, cuDF dtypes, or a mix of both

    """
    if isinstance(x, cudf._lib.scalar.Scalar):
        x = x.value
    return np.isnan(x)

def min_scalar_type(a):
    """
    For scalar a, returns the data type with the smallest size and smallest
    scalar kind which can hold its value. For non-scalar array a, returns
    the vector’s dtype unmodified.

    Parameters
    -------
    a : cuDF or NumPy scalar

    Returns
    -------
    result : cuDF dtype

    See Also
    -------
    numpy.mim_scalar_type

    Notes
    --------
    Accepts numpy dtypes, cuDF dtypes, or a mix of both

    """
    if isinstance(a, cudf.Scalar):
        a = a.value
    result = np.min_scalar_type(a)
    if result == np.dtype('float16'):
        return cudf.Float32Dtype()
    return cudf.dtype(result)

def promote_types(type1, type2):
    """
    Returns the data type with the smallest size and smallest scalar kind
    to which both type1 and type2 may be safely cast.

    Parameters
    -------
    type1 : cuDF or NumPy dtype
    type2 : cuDF or NumPy dtype

    Returns
    -------
    result : cuDF dtype, the promoted type

    See Also
    --------
    numpy.promote_types

    Notes
    --------
    Accepts numpy dtypes, cuDF dtypes, or a mix of both

    """
    if isinstance(type1, cudf.Generic):
        type1 = type1.numpy_dtype
    if isinstance(type2, cudf.Generic):
        type2 = type2.numpy_dtype

    result = np.promote_types(type1, type2)
    if result == np.dtype('float16'):
        return cudf.Float32Dtype()
    return cudf.dtype(result)

def isscalar(element):
    """
    Returns True if the type of `element` is a scalar type, 
    including cuDF, NumPy, and standard python scalars

    Parameters
    ----------
    element : any
        Input argument, can be of any type.
    Returns
    -------
    val : bool
        True if `element` is a scalar type, False if it is not.

    See Also
    --------
    numpy.isscalar

    """

    return isinstance(element, cudf._lib.scalar.Scalar) or np.isscalar(element)
