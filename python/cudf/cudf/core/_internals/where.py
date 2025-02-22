# Copyright (c) 2021-2025, NVIDIA CORPORATION.
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

import cudf
from cudf.api.types import _is_non_decimal_numeric_dtype, is_scalar
from cudf.core.dtypes import CategoricalDtype
from cudf.utils.dtypes import find_common_type, is_mixed_with_object_dtype

if TYPE_CHECKING:
    from cudf._typing import DtypeObj, ScalarLike
    from cudf.core.column import ColumnBase


def _normalize_categorical(input_col, other):
    if isinstance(input_col, cudf.core.column.CategoricalColumn):
        if cudf.api.types.is_scalar(other):
            try:
                other = input_col._encode(other)
            except ValueError:
                # When other is not present in categories,
                # fill with Null.
                other = None
            other = cudf.Scalar(other, dtype=input_col.codes.dtype)
        elif isinstance(other, cudf.core.column.CategoricalColumn):
            other = other.codes

        input_col = input_col.codes
    return input_col, other


def _check_and_cast_columns_with_other(
    source_col: ColumnBase,
    other: ScalarLike | ColumnBase,
    inplace: bool,
) -> tuple[ColumnBase, ScalarLike | ColumnBase]:
    # Returns type-casted `source_col` & `other` based on `inplace`.
    from cudf.core.column import as_column

    source_dtype = source_col.dtype
    if isinstance(source_dtype, CategoricalDtype):
        return _normalize_categorical(source_col, other)

    other_is_scalar = is_scalar(other)
    if other_is_scalar:
        if isinstance(other, (float, np.floating)) and not np.isnan(other):
            try:
                is_safe = source_dtype.type(other) == other
            except OverflowError:
                is_safe = False

            if not is_safe:
                raise TypeError(
                    f"Cannot safely cast non-equivalent "
                    f"{type(other).__name__} to {source_dtype.name}"
                )

        if cudf.utils.utils.is_na_like(other):
            return _normalize_categorical(
                source_col, cudf.Scalar(other, dtype=source_dtype)
            )

    mixed_err = (
        "cudf does not support mixed types, please type-cast the column of "
        "dataframe/series and other to same dtypes."
    )

    if inplace:
        other = cudf.Scalar(other) if other_is_scalar else other
        if is_mixed_with_object_dtype(other, source_col):
            raise TypeError(mixed_err)

        if not _can_cast(other.dtype, source_dtype):
            warnings.warn(
                f"Type-casting from {other.dtype} "
                f"to {source_dtype}, there could be potential data loss"
            )
        return _normalize_categorical(source_col, other.astype(source_dtype))

    if _is_non_decimal_numeric_dtype(source_dtype) and as_column(
        other
    ).can_cast_safely(source_dtype):
        common_dtype = source_dtype
    else:
        common_dtype = find_common_type(
            [
                source_dtype,
                np.min_scalar_type(other) if other_is_scalar else other.dtype,
            ]
        )

    if other_is_scalar:
        other = cudf.Scalar(other)

    if is_mixed_with_object_dtype(other, source_col) or (
        source_dtype.kind == "b" and common_dtype.kind != "b"
    ):
        raise TypeError(mixed_err)

    other = other.astype(common_dtype)

    return _normalize_categorical(source_col.astype(common_dtype), other)


def _can_cast(from_dtype: DtypeObj, to_dtype: DtypeObj) -> bool:
    """
    Utility function to determine if we can cast
    from `from_dtype` to `to_dtype`. This function primarily calls
    `np.can_cast` but with some special handling around
    cudf specific dtypes.
    """
    # TODO : Add precision & scale checking for
    # decimal types in future

    if isinstance(from_dtype, cudf.core.dtypes.DecimalDtype):
        if isinstance(to_dtype, cudf.core.dtypes.DecimalDtype):
            return True
        elif isinstance(to_dtype, np.dtype):
            if to_dtype.kind in {"i", "f", "u", "U", "O"}:
                return True
            else:
                return False
        else:
            return False
    elif isinstance(from_dtype, np.dtype):
        if isinstance(to_dtype, np.dtype):
            return np.can_cast(from_dtype, to_dtype)
        elif isinstance(to_dtype, cudf.core.dtypes.DecimalDtype):
            if from_dtype.kind in {"i", "f", "u", "U", "O"}:
                return True
            else:
                return False
        elif isinstance(to_dtype, cudf.CategoricalDtype):
            return True
        else:
            return False
    elif isinstance(from_dtype, cudf.ListDtype):
        # TODO: Add level based checks too once casting of
        # list columns is supported
        if isinstance(to_dtype, cudf.ListDtype):
            return np.can_cast(from_dtype.leaf_type, to_dtype.leaf_type)
        else:
            return False
    elif isinstance(from_dtype, cudf.CategoricalDtype):
        if isinstance(to_dtype, cudf.CategoricalDtype):
            return True
        elif isinstance(to_dtype, np.dtype):
            return np.can_cast(from_dtype.categories.dtype, to_dtype)
        else:
            return False
    else:
        return np.can_cast(from_dtype, to_dtype)
