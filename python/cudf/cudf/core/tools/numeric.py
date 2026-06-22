# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd

from cudf.core.column import as_column
from cudf.core.dtype.validators import (
    is_dtype_obj_numeric,
    is_dtype_obj_string,
)
from cudf.core.dtypes import CategoricalDtype, ListDtype, StructDtype
from cudf.core.index import Index, ensure_index
from cudf.core.series import Series
from cudf.utils.dtypes import (
    can_convert_to_column,
)

if TYPE_CHECKING:
    from cudf.core.column.numerical import NumericalColumn
    from cudf.core.column.string import StringColumn


def to_numeric(
    arg,
    errors: Literal["raise", "coerce"] = "raise",
    downcast: Literal["integer", "signed", "unsigned", "float"] | None = None,
    dtype_backend=None,
):
    """
    Convert argument into numerical types.

    Parameters
    ----------
    arg : column-convertible
        The object to convert to numeric types
    errors : {'raise', 'coerce'}, defaults 'raise'
        Policy to handle errors during parsing.

        * 'raise' will notify user all errors encountered.
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
    Series, Index or ndarray
        Depending on the input, a Series is returned for Series input, an
        Index for Index input, otherwise an ndarray.

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
    >>> import warnings
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore", UserWarning)
    ...     ser = cudf.to_numeric(s, downcast='signed')
    >>> ser
    0       1
    1       2
    2    3000
    dtype: int16
    >>> s = cudf.Series(['apple', '1.0', '3e3'])
    >>> cudf.to_numeric(s, errors='coerce')
    0       NaN
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
    if errors not in {"raise", "coerce"}:
        raise ValueError("invalid error value specified")

    if downcast not in {None, "integer", "signed", "unsigned", "float"}:
        raise ValueError("invalid downcasting method provided")

    if not can_convert_to_column(arg) or (
        hasattr(arg, "ndim") and arg.ndim > 1
    ):
        raise ValueError("arg must be column convertible")

    col = as_column(arg)
    dtype = col.dtype

    # A pandas masked (nullable) string dtype should produce a masked numeric
    # result (``Int64``/``Float64``), preserving nulls as ``pd.NA``.
    nullable = isinstance(dtype, pd.StringDtype) and dtype.na_value is pd.NA

    if dtype.kind in "mM":
        if isinstance(arg, np.ndarray) and arg.dtype.kind in "mM":
            # pandas returns the raw integer view at the array's native
            # resolution. cudf may have upcast an unsupported resolution
            # (e.g. ``datetime64[D]`` -> ``datetime64[s]``), which would change
            # the integer values, so view the original data instead.
            col = as_column(arg.view(np.dtype(np.int64)))
        else:
            col = col.astype(np.dtype(np.int64))
    elif isinstance(dtype, CategoricalDtype):
        cat_dtype = col.dtype.categories.dtype  # type: ignore[union-attr]
        if cat_dtype.kind in "iufb":
            if cat_dtype.kind in "iub" and col.has_nulls():
                # pandas promotes a null-containing categorical with
                # integer/bool categories to float, since those categories
                # cannot represent the missing value.
                col = col.astype(np.dtype(np.float64))
            else:
                col = col.astype(cat_dtype)
        else:
            col = _convert_str_col(
                col._get_decategorized_column(),  # type: ignore[attr-defined]
                errors,
                downcast,
            )
    elif is_dtype_obj_string(dtype):
        col = _convert_str_col(col, errors, downcast, nullable=nullable)  # type: ignore[arg-type]
    elif isinstance(dtype, (ListDtype, StructDtype)):
        raise ValueError("Input does not support nested datatypes")
    elif is_dtype_obj_numeric(dtype, include_decimal=False):
        pass
    else:
        raise ValueError("Unrecognized datatype")

    # str->float conversion may require lower precision
    if col.dtype == np.dtype(np.float32):
        col = col.astype(np.dtype(np.float64))

    if downcast:
        if downcast == "float":
            # we support only float32 & float64
            type_set = [
                np.dtype(np.float32).char,
                np.dtype(np.float64).char,
            ]
        elif downcast in ("integer", "signed"):
            type_set = list(np.typecodes["Integer"])  # type: ignore[arg-type]
        elif downcast == "unsigned":
            type_set = list(np.typecodes["UnsignedInteger"])  # type: ignore[arg-type]

        for t in type_set:
            downcast_dtype: np.dtype = np.dtype(t)
            if downcast_dtype.itemsize <= col.dtype.itemsize:
                if col.can_cast_safely(downcast_dtype):
                    if (
                        downcast == "float"
                        and downcast_dtype.kind == "f"
                        and not _float_downcast_preserves_value(
                            col, downcast_dtype
                        )
                    ):
                        # pandas only narrows a float when every value survives
                        # the round-trip; otherwise it keeps the wider float.
                        continue
                    col = col.cast(downcast_dtype)
                    break

    if isinstance(arg, (Series, pd.Series)):
        return Series._from_column(
            col, name=arg.name, index=ensure_index(arg.index)
        )
    elif isinstance(arg, (Index, pd.Index)):
        # pandas returns an Index (preserving the name) for Index input.
        return Index._from_column(col, name=arg.name)
    else:
        if col.has_nulls():
            # To match pandas, always return a floating type filled with nan.
            col = col.astype(np.dtype(np.float64)).fillna(np.nan)
        return col.values


def _float_downcast_preserves_value(col, to_dtype: np.dtype) -> bool:
    """Return True if narrowing ``col`` to the float ``to_dtype`` is lossless.

    Mirrors ``pandas.core.dtypes.cast.maybe_downcast_numeric`` for floats: the
    cast is accepted only when every value round-trips within an absolute
    tolerance that depends on the target float width (and NaN/NaN aligns).
    """
    f64 = np.dtype(np.float64)
    base = col.astype(f64)
    round_tripped = base.astype(to_dtype).astype(f64)
    # pandas size_tols: float32 -> 5e-4, float64 -> 5e-8, float128 -> 5e-16
    atol = {4: 5e-4, 8: 5e-8, 16: 5e-16}.get(to_dtype.itemsize, 0.0)
    diff = round_tripped - base
    within = (diff <= atol) & (diff >= -atol)
    # ``equal`` covers exact round-trips, including +/-inf where ``diff`` would
    # be NaN; ``both_nan`` matches NaN positions (as pandas does with
    # ``equal_nan=True``).
    equal = round_tripped == base
    both_nan = round_tripped.isnan() & base.isnan()
    return bool((within | equal | both_nan).fillna(True).all())


def _convert_nullable_str_col(
    string_col: StringColumn,
    errors: Literal["raise", "coerce"],
) -> NumericalColumn:
    """Convert a pandas masked (nullable) string column to masked numeric.

    Produces an ``Int64`` column when every parseable value is an integer and a
    ``Float64`` column otherwise, preserving the null mask as ``pd.NA``.
    """
    if errors == "coerce":
        # Null out values that cannot be parsed as a number.
        invalid = string_col.is_float().unary_operator("not")
        string_col = string_col.copy()
        string_col[invalid.fillna(False)] = None

    if string_col.is_all_integer():
        return string_col.astype(np.dtype(np.int64)).astype(  # type: ignore[return-value]
            pd.Int64Dtype()
        )

    converted_col = (
        string_col.to_lower()
        .find_and_replace(as_column("", length=1), as_column("NaN", length=1))
        .replace_multiple(
            as_column(["+", "inf", "inity"]),  # type: ignore[arg-type]
            as_column(["", "Inf", ""]),  # type: ignore[arg-type]
        )
    )
    if not converted_col.is_float().all():
        if errors != "coerce":
            raise ValueError("Unable to convert some strings to numerics.")
        non_numerics = converted_col.is_float().unary_operator("not")
        converted_col[non_numerics.fillna(False)] = None
    return converted_col.astype(np.dtype(np.float64)).astype(  # type: ignore[return-value]
        pd.Float64Dtype()
    )


def _convert_str_col(
    col: StringColumn,
    errors: Literal["raise", "coerce"],
    downcast: Literal["integer", "signed", "unsigned", "float"] | None = None,
    nullable: bool = False,
) -> NumericalColumn:
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
    errors : {'raise', 'coerce'}, same as ``to_numeric``
    downcast : Same as ``to_numeric``, see description for use
    nullable : If True, the input came from a pandas masked (nullable) string
        dtype and the result should be a masked ``Int64``/``Float64`` column.

    Returns
    -------
    Converted numeric column
    """
    if not is_dtype_obj_string(col.dtype):
        raise TypeError("col must be string dtype.")

    string_col = cast("StringColumn", col)

    if nullable:
        return _convert_nullable_str_col(string_col, errors)

    if string_col.is_all_integer():
        if string_col.is_all_integer(np.dtype(np.int64)):
            return col.astype(dtype=np.dtype(np.int64))  # type: ignore[return-value]
        elif string_col.is_all_integer(np.dtype(np.uint64)):
            return col.astype(dtype=np.dtype(np.uint64))  # type: ignore[return-value]
        elif errors == "coerce":
            # The value overflows uint64; pandas represents such a result as a
            # float, so match that here.
            return col.astype(dtype=np.dtype(np.float64))  # type: ignore[return-value]
        else:
            # pandas returns an object array of Python ints in this case. cudf
            # has no object numeric type, so surface the overflow (under
            # cudf.pandas this triggers a fallback to pandas, which produces
            # the object result).
            raise OverflowError(
                "Integer string value is out of bounds for int64/uint64."
            )

    # TODO: This can be handled by libcudf in
    # future see StringColumn.as_numerical_column
    converted_col = (
        col.to_lower()
        .find_and_replace(as_column("", length=1), as_column("NaN", length=1))
        .replace_multiple(
            as_column(["+", "inf", "inity"]),  # type: ignore[arg-type]
            as_column(["", "Inf", ""]),  # type: ignore[arg-type]
        )
    )

    is_float = converted_col.is_float()
    if is_float.all():
        if downcast in {"unsigned", "signed", "integer"}:
            warnings.warn(
                UserWarning(
                    "Downcasting from float to int will be "
                    "limited by float32 precision."
                )
            )
            return converted_col.astype(dtype=np.dtype(np.float32))  # type: ignore[return-value]
        else:
            return converted_col.astype(dtype=np.dtype(np.float64))  # type: ignore[return-value]
    else:
        if errors == "coerce":
            non_numerics = is_float.unary_operator("not")
            converted_col[non_numerics] = None
            converted_col = converted_col.astype(np.dtype(np.float64))  # type: ignore[assignment]
            return converted_col  # type: ignore[return-value]
        else:
            raise ValueError("Unable to convert some strings to numerics.")
