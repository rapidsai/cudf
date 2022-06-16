# Copyright (c) 2021-2022, NVIDIA CORPORATION.

import warnings
from typing import Any, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

import cudf
from cudf._typing import ColumnLike, ScalarLike
from cudf.core.column import ColumnBase
from cudf.core.dataframe import DataFrame
from cudf.core.frame import Frame
from cudf.core.index import Index
from cudf.core.missing import NA
from cudf.core.series import Series
from cudf.core.single_column_frame import SingleColumnFrame


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


def where(
    frame: Frame,
    cond: Any,
    other: Any = None,
    inplace: bool = False,
) -> Optional[Union[Frame]]:
    """
    Replace values where the condition is False.

    Parameters
    ----------
    cond : bool Series/DataFrame, array-like
        Where cond is True, keep the original value.
        Where False, replace with corresponding value from other.
        Callables are not supported.
    other: scalar, list of scalars, Series/DataFrame
        Entries where cond is False are replaced with
        corresponding value from other. Callables are not
        supported. Default is None.

        DataFrame expects only Scalar or array like with scalars or
        dataframe with same dimension as frame.

        Series expects only scalar or series like with same length
    inplace : bool, default False
        Whether to perform the operation in place on the data.

    Returns
    -------
    Same type as caller

    Examples
    --------
    >>> import cudf
    >>> df = DataFrame({"A":[1, 4, 5], "B":[3, 5, 8]})
    >>> df.where(df % 2 == 0, [-1, -1])
       A  B
    0 -1 -1
    1  4 -1
    2 -1  8

    >>> ser = Series([4, 3, 2, 1, 0])
    >>> ser.where(ser > 2, 10)
    0     4
    1     3
    2    10
    3    10
    4    10
    dtype: int64
    >>> ser.where(ser > 2)
    0       4
    1       3
    2    <NA>
    3    <NA>
    4    <NA>
    dtype: int64
    """

    if isinstance(frame, DataFrame):
        if hasattr(cond, "__cuda_array_interface__"):
            if isinstance(cond, Series):
                cond = DataFrame(
                    {name: cond for name in frame._column_names},
                    index=frame.index,
                )
            else:
                cond = DataFrame(
                    cond, columns=frame._column_names, index=frame.index
                )
        elif (
            hasattr(cond, "__array_interface__")
            and cond.__array_interface__["shape"] != frame.shape
        ):
            raise ValueError("conditional must be same shape as self")
        elif not isinstance(cond, DataFrame):
            cond = frame.from_pandas(pd.DataFrame(cond))

        common_cols = set(frame._column_names).intersection(
            set(cond._column_names)
        )
        if len(common_cols) > 0:
            # If `frame` and `cond` are having unequal index,
            # then re-index `cond`.
            if not frame.index.equals(cond.index):
                cond = cond.reindex(frame.index)
        else:
            if cond.shape != frame.shape:
                raise ValueError(
                    """Array conditional must be same shape as self"""
                )
            # Setting `frame` column names to `cond`
            # as `cond` has no column names.
            cond._set_column_names_like(frame)

        (
            source_df,
            others,
        ) = _normalize_columns_and_scalars_type(frame, other)
        if isinstance(others, Frame):
            others = others._data.columns

        out_df = DataFrame(index=frame.index)
        if len(frame._columns) != len(others):
            raise ValueError(
                """Replacement list length or number of dataframe columns
                should be equal to Number of columns of dataframe"""
            )
        for i, column_name in enumerate(frame._column_names):
            input_col = source_df._data[column_name]
            other_column = others[i]
            if column_name in cond._data:
                if isinstance(input_col, cudf.core.column.CategoricalColumn):
                    if cudf.api.types.is_scalar(other_column):
                        try:
                            other_column = input_col._encode(other_column)
                        except ValueError:
                            # When other is not present in categories,
                            # fill with Null.
                            other_column = None
                        other_column = cudf.Scalar(
                            other_column, dtype=input_col.codes.dtype
                        )
                    elif isinstance(
                        other_column, cudf.core.column.CategoricalColumn
                    ):
                        other_column = other_column.codes
                    input_col = input_col.codes

                result = cudf._lib.copying.copy_if_else(
                    input_col, other_column, cond._data[column_name]
                )

                if isinstance(
                    frame._data[column_name],
                    cudf.core.column.CategoricalColumn,
                ):
                    result = cudf.core.column.build_categorical_column(
                        categories=frame._data[column_name].categories,
                        codes=cudf.core.column.build_column(
                            result.base_data, dtype=result.dtype
                        ),
                        mask=result.base_mask,
                        size=result.size,
                        offset=result.offset,
                        ordered=frame._data[column_name].ordered,
                    )
            else:
                out_mask = cudf._lib.null_mask.create_null_mask(
                    len(input_col),
                    state=cudf._lib.null_mask.MaskState.ALL_NULL,
                )
                result = input_col.set_mask(out_mask)
            out_df[column_name] = frame[column_name].__class__(result)

        return frame._mimic_inplace(out_df, inplace=inplace)

    else:
        frame = cast(SingleColumnFrame, frame)
        if isinstance(other, DataFrame):
            raise NotImplementedError(
                "cannot align with a higher dimensional Frame"
            )
        input_col = frame._data[frame.name]
        cond = cudf.core.column.as_column(cond)
        if len(cond) != len(frame):
            raise ValueError(
                """Array conditional must be same shape as self"""
            )

        (
            input_col,
            other,
        ) = _normalize_columns_and_scalars_type(frame, other, inplace)

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

        result = cudf._lib.copying.copy_if_else(input_col, other, cond)

        if isinstance(
            frame._data[frame.name], cudf.core.column.CategoricalColumn
        ):
            result = cudf.core.column.build_categorical_column(
                categories=cast(
                    cudf.core.column.CategoricalColumn,
                    frame._data[frame.name],
                ).categories,
                codes=cudf.core.column.build_column(
                    result.base_data, dtype=result.dtype
                ),
                mask=result.base_mask,
                size=result.size,
                offset=result.offset,
                ordered=cast(
                    cudf.core.column.CategoricalColumn,
                    frame._data[frame.name],
                ).ordered,
            )

        if isinstance(frame, Index):
            result = Index(result, name=frame.name)
        else:
            result = frame._from_data({frame.name: result}, frame._index)

        return frame._mimic_inplace(result, inplace=inplace)
