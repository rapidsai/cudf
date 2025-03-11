# Copyright (c) 2021-2025, NVIDIA CORPORATION.
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

import cudf
from cudf.api.types import is_scalar
from cudf.core.dtypes import CategoricalDtype
from cudf.core.scalar import pa_scalar_to_plc_scalar
from cudf.utils.dtypes import (
    CUDF_STRING_DTYPE,
    cudf_dtype_to_pa_type,
    find_common_type,
    is_dtype_obj_numeric,
    is_mixed_with_object_dtype,
)

if TYPE_CHECKING:
    import pylibcudf as plc

    from cudf._typing import ScalarLike
    from cudf.core.column import ColumnBase


def _check_and_cast_columns_with_other(
    source_col: ColumnBase,
    other: ScalarLike | ColumnBase,
    inplace: bool,
) -> tuple[ColumnBase, plc.Scalar | ColumnBase]:
    # Returns type-casted `source_col` & `other` based on `inplace`.
    from cudf.core.column import as_column

    source_dtype = source_col.dtype
    other_is_scalar = is_scalar(other)

    if isinstance(source_dtype, CategoricalDtype):
        if other_is_scalar:
            try:
                other = source_col._encode(other)  # type: ignore[attr-defined]
            except ValueError:
                # When other is not present in categories,
                # fill with Null.
                other = None
            other = pa_scalar_to_plc_scalar(
                pa.scalar(
                    other,
                    type=cudf_dtype_to_pa_type(source_col.codes.dtype),  # type: ignore[attr-defined]
                )
            )
        elif isinstance(other.dtype, CategoricalDtype):
            other = other.codes  # type: ignore[union-attr]

        return source_col.codes, other  # type: ignore[attr-defined]

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
            return source_col, pa_scalar_to_plc_scalar(
                pa.scalar(None, type=cudf_dtype_to_pa_type(source_dtype))
            )

    mixed_err = (
        "cudf does not support mixed types, please type-cast the column of "
        "dataframe/series and other to same dtypes."
    )

    if inplace:
        other_col = as_column(other)
        if is_mixed_with_object_dtype(other_col, source_col):
            raise TypeError(mixed_err)

        if other_col.dtype != source_dtype:
            try:
                warn = (
                    find_common_type((other_col.dtype, source_dtype))
                    == CUDF_STRING_DTYPE
                )
            except NotImplementedError:
                warn = True
            if warn:
                warnings.warn(
                    f"Type-casting from {other_col.dtype} "
                    f"to {source_dtype}, there could be potential data loss"
                )
        if other_is_scalar:
            other_out = pa_scalar_to_plc_scalar(
                pa.scalar(other, type=cudf_dtype_to_pa_type(source_dtype))
            )
        else:
            other_out = other_col.astype(source_dtype)
        return source_col, other_out

    if is_dtype_obj_numeric(source_dtype, include_decimal=False) and as_column(
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

    if is_mixed_with_object_dtype(as_column(other), source_col) or (
        source_dtype.kind == "b" and common_dtype.kind != "b"
    ):
        raise TypeError(mixed_err)

    if other_is_scalar:
        other_out = pa_scalar_to_plc_scalar(
            pa.scalar(other, type=cudf_dtype_to_pa_type(common_dtype))
        )
    else:
        other_out = other.astype(common_dtype)

    return source_col.astype(common_dtype), other_out
