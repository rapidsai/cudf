# Copyright (c) 2021-2022, NVIDIA CORPORATION.

import warnings
from typing import Any, Tuple, Union

import numpy as np

import cudf
from cudf._typing import ColumnLike, ScalarLike
from cudf.core.column import ColumnBase
from cudf.core.dataframe import DataFrame
from cudf.core.frame import Frame
from cudf.core.index import Index
from cudf.core.missing import NA
from cudf.core.series import Series


def _normalize_scalars(col: ColumnBase, other: ScalarLike) -> ScalarLike:
    """
    Try to normalize scalar values as per col dtype
    """
    if (isinstance(other, float) and not np.isnan(other)) and (
        col.dtype.type(other) != other
    ):
        raise TypeError(
            f"Cannot safely cast non-equivalent "
            f"{type(other).__name__} to {col.dtype.name}"
        )

    return cudf.Scalar(other, dtype=col.dtype if other in {None, NA} else None)


def _check_and_cast_columns_with_other(
    source_col: ColumnBase,
    other: Union[ScalarLike, ColumnBase],
    inplace: bool,
) -> Tuple[ColumnBase, Union[ScalarLike, ColumnBase]]:
    """
    Returns type-casted column `source_col` & scalar `other_scalar`
    based on `inplace` parameter.
    """
    if cudf.api.types.is_categorical_dtype(source_col.dtype):
        return source_col, other

    if cudf.api.types.is_scalar(other):
        device_obj = _normalize_scalars(source_col, other)
    else:
        device_obj = other

    if other is None:
        return source_col, device_obj
    elif cudf.utils.dtypes.is_mixed_with_object_dtype(device_obj, source_col):
        raise TypeError(
            "cudf does not support mixed types, please type-cast "
            "the column of dataframe/series and other "
            "to same dtypes."
        )
    if inplace:
        if not cudf.utils.dtypes._can_cast(device_obj.dtype, source_col.dtype):
            warnings.warn(
                f"Type-casting from {device_obj.dtype} "
                f"to {source_col.dtype}, there could be potential data loss"
            )
        return source_col, device_obj.astype(source_col.dtype)
    else:
        if (
            cudf.api.types.is_scalar(other)
            and cudf.api.types._is_non_decimal_numeric_dtype(source_col.dtype)
            and cudf.utils.dtypes._can_cast(other, source_col.dtype)
        ):
            common_dtype = source_col.dtype
            return (
                source_col.astype(common_dtype),
                cudf.Scalar(other, dtype=common_dtype),
            )
        else:
            common_dtype = cudf.utils.dtypes.find_common_type(
                [
                    source_col.dtype,
                    np.min_scalar_type(other)
                    if cudf.api.types.is_scalar(other)
                    else other.dtype,
                ]
            )
            if cudf.api.types.is_scalar(device_obj):
                device_obj = cudf.Scalar(other, dtype=common_dtype)
            else:
                device_obj = device_obj.astype(common_dtype)
            return source_col.astype(common_dtype), device_obj


def _normalize_columns_and_scalars_type(
    frame: Frame,
    other: Any,
    inplace: bool = False,
) -> Tuple[Union[Frame, ColumnLike], Any]:
    """
    Try to normalize the other's dtypes as per frame.

    Parameters
    ----------

    frame : Can be a DataFrame or Series or Index
    other : Can be a DataFrame, Series, Index, Array
        like object or a scalar value

        if frame is DataFrame, other can be only a
        scalar or array like with size of number of columns
        in DataFrame or a DataFrame with same dimension

        if frame is Series, other can be only a scalar or
        a series like with same length as frame

    Returns:
    --------
    A dataframe/series/list/scalar form of normalized other
    """
    if isinstance(frame, DataFrame) and isinstance(other, DataFrame):
        source_df = frame.copy(deep=False)
        other_df = other.copy(deep=False)
        for self_col in source_df._column_names:
            source_col, other_col = _check_and_cast_columns_with_other(
                source_col=source_df._data[self_col],
                other=other_df._data[self_col],
                inplace=inplace,
            )
            source_df._data[self_col] = source_col
            other_df._data[self_col] = other_col
        return source_df, other_df

    elif isinstance(frame, (Series, Index)) and not cudf.api.types.is_scalar(
        other
    ):
        other = cudf.core.column.as_column(other)
        input_col = frame._data[frame.name]
        return _check_and_cast_columns_with_other(
            source_col=input_col, other=other, inplace=inplace
        )
    else:
        # Handles scalar or list/array like scalars
        if isinstance(frame, (Series, Index)) and cudf.api.types.is_scalar(
            other
        ):
            input_col = frame._data[frame.name]
            return _check_and_cast_columns_with_other(
                source_col=frame._data[frame.name],
                other=other,
                inplace=inplace,
            )

        elif isinstance(frame, DataFrame):
            source_df = frame.copy(deep=False)
            others = []
            for i, col_name in enumerate(frame._column_names):
                (
                    source_col,
                    other_scalar,
                ) = _check_and_cast_columns_with_other(
                    source_col=source_df._data[col_name],
                    other=other
                    if cudf.api.types.is_scalar(other)
                    else other[i],
                    inplace=inplace,
                )
                source_df._data[col_name] = source_col
                others.append(other_scalar)
            return source_df, others
        else:
            raise ValueError(
                f"Inappropriate input {type(frame)} "
                f"and other {type(other)} combination"
            )


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


def _make_categorical_like(result, column):
    if isinstance(column, cudf.core.column.CategoricalColumn):
        result = cudf.core.column.build_categorical_column(
            categories=column.categories,
            codes=cudf.core.column.build_column(
                result.base_data, dtype=result.dtype
            ),
            mask=result.base_mask,
            size=result.size,
            offset=result.offset,
            ordered=column.ordered,
        )
    return result
