# Copyright (c) 2018-2024, NVIDIA CORPORATION.
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import cudf
from cudf import _lib as libcudf
from cudf._lib import strings as libstrings
from cudf.api.types import _is_non_decimal_numeric_dtype, is_string_dtype
from cudf.core.column import as_column
from cudf.core.dtypes import CategoricalDtype
from cudf.core.index import ensure_index
from cudf.utils.dtypes import can_convert_to_column

if TYPE_CHECKING:
    from cudf.core.column import ColumnBase


def to_numeric(arg, errors="raise", downcast=None, dtype_backend=None):
    """
    Convert argument into numerical types.

    Parameters
    ----------
    arg : column-convertible
        The object to convert to numeric types
    errors : {'raise', 'ignore', 'coerce'}, defaults 'raise'
        Policy to handle errors during parsing.

        * 'raise' will notify user all errors encountered.
        * 'ignore' will skip error and returns ``arg``.
        * 'coerce' will leave invalid values as nulls.
    downcast : {'integer', 'signed', 'unsigned', 'float'}, defaults None
        If set, will try to down-convert the datatype of the
        parsed results to smallest possible type. For each `downcast`
        type, this method will determine the smallest possible
        dtype from the following sets:

        * {'integer', 'signed'}: all integer types greater or equal to
          `np.int8`
        * {'unsigned'}: all unsigned types greater or equal to `np.uint8`
        * {'float'}: all floating types greater or equal to `np.float32`

        Note that downcast behavior is decoupled from parsing. Errors
        encountered during downcast is raised regardless of ``errors``
        parameter.
    dtype_backend : None
        Not implemented.

    Returns
    -------
    Series or ndarray
        Depending on the input, if series is passed in, series is returned,
        otherwise ndarray

    Examples
    --------
    >>> s = cudf.Series(['1', '2.0', '3e3'])
    >>> cudf.to_numeric(s)
    0       1.0
    1       2.0
    2    3000.0
    dtype: float64
    >>> cudf.to_numeric(s, downcast='float')
    0       1.0
    1       2.0
    2    3000.0
    dtype: float32
    >>> cudf.to_numeric(s, downcast='signed')
    0       1
    1       2
    2    3000
    dtype: int16
    >>> s = cudf.Series(['apple', '1.0', '3e3'])
    >>> cudf.to_numeric(s, errors='ignore')
    0    apple
    1      1.0
    2      3e3
    dtype: object
    >>> cudf.to_numeric(s, errors='coerce')
    0      <NA>
    1       1.0
    2    3000.0
    dtype: float64

    .. pandas-compat::
        :func:`pandas.to_numeric`

        An important difference from pandas is that this function does not
        accept mixed numeric/non-numeric type sequences.
        For example ``[1, 'a']``. A ``TypeError`` will be raised when such
        input is received, regardless of ``errors`` parameter.
    """
    if dtype_backend is not None:
        raise NotImplementedError(
            "dtype_backend is not currently implemented."
        )
    if errors not in {"raise", "ignore", "coerce"}:
        raise ValueError("invalid error value specified")
    elif errors == "ignore":
        warnings.warn(
            "errors='ignore' is deprecated and will raise in "
            "a future version. Use to_numeric without passing `errors` "
            "and catch exceptions explicitly instead",
            FutureWarning,
        )

    if downcast not in {None, "integer", "signed", "unsigned", "float"}:
        raise ValueError("invalid downcasting method provided")

    if not can_convert_to_column(arg) or (
        hasattr(arg, "ndim") and arg.ndim > 1
    ):
        raise ValueError("arg must be column convertible")

    col = as_column(arg)
    dtype = col.dtype

    if dtype.kind in "mM":
        col = col.astype(cudf.dtype("int64"))
    elif isinstance(dtype, CategoricalDtype):
        cat_dtype = col.dtype.type
        if _is_non_decimal_numeric_dtype(cat_dtype):
            col = col.astype(cat_dtype)
        else:
            try:
                col = _convert_str_col(
                    col._get_decategorized_column(), errors, downcast
                )
            except ValueError as e:
                if errors == "ignore":
                    return arg
                else:
                    raise e
    elif is_string_dtype(dtype):
        try:
            col = _convert_str_col(col, errors, downcast)
        except ValueError as e:
            if errors == "ignore":
                return arg
            else:
                raise e
    elif isinstance(dtype, (cudf.ListDtype, cudf.StructDtype)):
        raise ValueError("Input does not support nested datatypes")
    elif _is_non_decimal_numeric_dtype(dtype):
        pass
    else:
        raise ValueError("Unrecognized datatype")

    # str->float conversion may require lower precision
    if col.dtype == cudf.dtype("float32"):
        col = col.astype("float64")

    if downcast:
        if downcast == "float":
            # we support only float32 & float64
            type_set = [
                cudf.dtype(np.float32).char,
                cudf.dtype(np.float64).char,
            ]
        elif downcast in ("integer", "signed"):
            type_set = list(np.typecodes["Integer"])
        elif downcast == "unsigned":
            type_set = list(np.typecodes["UnsignedInteger"])

        for t in type_set:
            downcast_dtype = cudf.dtype(t)
            if downcast_dtype.itemsize <= col.dtype.itemsize:
                if col.can_cast_safely(downcast_dtype):
                    col = libcudf.unary.cast(col, downcast_dtype)
                    break

    if isinstance(arg, (cudf.Series, pd.Series)):
        return cudf.Series._from_column(
            col, name=arg.name, index=ensure_index(arg.index)
        )
    else:
        if col.has_nulls():
            # To match pandas, always return a floating type filled with nan.
            col = col.astype(float).fillna(np.nan)
        return col.values


def _convert_str_col(col, errors, _downcast=None):
    """
    Converts a string column to numeric column

    Converts to integer column if all strings are integer-like (isinteger.all)
    Otherwise, converts to float column if all strings are float-like (
    isfloat.all)

    If error == 'coerce', fill non-numerics strings with null

    Looks ahead to ``downcast`` parameter, if the float may be casted to
    integer, then only process in float32 pipeline.

    Parameters
    ----------
    col : The string column to convert, must be string dtype
    errors : {'raise', 'ignore', 'coerce'}, same as ``to_numeric``
    _downcast : Same as ``to_numeric``, see description for use

    Returns
    -------
    Converted numeric column
    """
    if not is_string_dtype(col):
        raise TypeError("col must be string dtype.")

    is_integer = libstrings.is_integer(col)
    if is_integer.all():
        return col.astype(dtype=cudf.dtype("i8"))

    col = _proc_inf_empty_strings(col)

    is_float = libstrings.is_float(col)
    if is_float.all():
        if _downcast in {"unsigned", "signed", "integer"}:
            warnings.warn(
                UserWarning(
                    "Downcasting from float to int will be "
                    "limited by float32 precision."
                )
            )
            return col.astype(dtype=cudf.dtype("float32"))
        else:
            return col.astype(dtype=cudf.dtype("float64"))
    else:
        if errors == "coerce":
            col = libcudf.string_casting.stod(col)
            non_numerics = is_float.unary_operator("not")
            col[non_numerics] = None
            return col
        else:
            raise ValueError("Unable to convert some strings to numerics.")


def _proc_inf_empty_strings(col: ColumnBase) -> ColumnBase:
    """Handles empty and infinity strings"""
    col = libstrings.to_lower(col)
    col = col.find_and_replace(as_column([""]), as_column(["NaN"]))
    # TODO: This can be handled by libcudf in
    # future see StringColumn.as_numerical_column
    col = libstrings.replace_multi(
        col,
        as_column(["+", "inf", "inity"]),
        as_column(["", "Inf", ""]),
    )
    return col
