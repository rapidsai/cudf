# Copyright (c) 2021, NVIDIA CORPORATION.

import warnings
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd

import cudf
from cudf._typing import ScalarLike


def _normalize_scalars(
    col: cudf.core.column.ColumnBase, other: ScalarLike
) -> cudf.Scalar:
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

    return cudf.Scalar(other, dtype=col.dtype if other is None else None)


def _check_and_cast_columns(
    source_col: cudf.core.column.ColumnBase,
    other_col: cudf.core.column.ColumnBase,
    inplace: bool,
) -> Tuple[cudf.core.column.ColumnBase, cudf.core.column.ColumnBase]:
    """
    Returns type-casted columns of `source_col` & `other_col`
    based on `inplace` parameter.
    """
    if cudf.utils.dtypes.is_categorical_dtype(source_col.dtype):
        return source_col, other_col
    elif cudf.utils.dtypes.is_mixed_with_object_dtype(source_col, other_col):
        raise TypeError(
            "cudf does not support mixed types, please type-cast "
            "the column of dataframe/series and other "
            "to same dtypes."
        )
    if inplace:
        if not source_col.can_cast_safely(other_col.dtype):
            warnings.warn(
                f"Type-casting from {other_col.dtype} "
                f"to {source_col.dtype}, there could be potential data loss"
            )
        return source_col, other_col.astype(source_col.dtype)
    else:
        common_dtype = cudf.utils.dtypes.find_common_type(
            [source_col.dtype, other_col.dtype]
        )
        return source_col.astype(common_dtype), other_col.astype(common_dtype)


def _check_and_cast_columns_with_scalar(
    source_col: cudf.core.column.ColumnBase,
    other_scalar: ScalarLike,
    inplace: bool,
) -> Tuple[cudf.core.column.ColumnBase, ScalarLike]:
    """
    Returns type-casted column `source_col` & scalar `other_scalar`
    based on `inplace` parameter.
    """
    if cudf.utils.dtypes.is_categorical_dtype(source_col.dtype):
        return source_col, other_scalar

    device_scalar = _normalize_scalars(source_col, other_scalar)

    if other_scalar is None:
        return source_col, device_scalar
    elif cudf.utils.dtypes.is_mixed_with_object_dtype(
        device_scalar, source_col
    ):
        raise TypeError(
            "cudf does not support mixed types, please type-cast "
            "the column of dataframe/series and other "
            "to same dtypes."
        )
    if inplace:
        if not cudf.utils.dtypes.can_cast(
            device_scalar.dtype, source_col.dtype
        ):
            warnings.warn(
                f"Type-casting from {device_scalar.dtype} "
                f"to {source_col.dtype}, there could be potential data loss"
            )
        return source_col, device_scalar.astype(source_col.dtype)
    else:
        if pd.api.types.is_numeric_dtype(
            source_col.dtype
        ) and cudf.utils.dtypes.can_cast(other_scalar, source_col.dtype):
            common_dtype = source_col.dtype
        else:
            common_dtype = cudf.utils.dtypes.find_common_type(
                [source_col.dtype, np.min_scalar_type(other_scalar)]
            )

        source_col = source_col.astype(common_dtype)
        return source_col, cudf.Scalar(other_scalar, dtype=common_dtype)


def _normalize_columns_and_scalars_type(
    frame: cudf.core.frame.Frame, other: Any, inplace: bool = False
) -> Tuple[Union[cudf.core.frame.Frame, cudf.core.column.ColumnBase], Any]:
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
    if isinstance(frame, cudf.DataFrame) and isinstance(other, cudf.DataFrame):
        source_df = frame.copy(deep=False)
        other_df = other.copy(deep=False)
        for self_col in source_df._data.names:
            source_col, other_col = _check_and_cast_columns(
                source_col=source_df._data[self_col],
                other_col=other_df._data[self_col],
                inplace=inplace,
            )
            source_df._data[self_col] = source_col
            other_df._data[self_col] = other_col
        return source_df, other_df

    elif isinstance(
        frame, (cudf.Series, cudf.Index)
    ) and not cudf.utils.dtypes.is_scalar(other):
        other = cudf.core.column.as_column(other)
        input_col = frame._data[frame.name]
        return _check_and_cast_columns(
            source_col=input_col, other_col=other, inplace=inplace
        )
    else:
        # Handles scalar or list/array like scalars
        if isinstance(
            frame, (cudf.Series, cudf.Index)
        ) and cudf.utils.dtypes.is_scalar(other):
            input_col = frame._data[frame.name]
            return _check_and_cast_columns_with_scalar(
                source_col=frame._data[frame.name],
                other_scalar=other,
                inplace=inplace,
            )

        elif isinstance(frame, cudf.DataFrame):
            if cudf.utils.dtypes.is_scalar(other):
                other = [other for i in range(len(frame._data.names))]

            source_df = frame.copy(deep=False)
            others = []
            for col_name, other_sclr in zip(frame._data.names, other):

                (
                    source_col,
                    other_scalar,
                ) = _check_and_cast_columns_with_scalar(
                    source_col=source_df._data[col_name],
                    other_scalar=other_sclr,
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


def where(frame, cond, other=None, inplace=False):
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
    >>> df = cudf.DataFrame({"A":[1, 4, 5], "B":[3, 5, 8]})
    >>> df.where(df % 2 == 0, [-1, -1])
        A  B
    0 -1 -1
    1  4 -1
    2 -1  8

    >>> ser = cudf.Series([4, 3, 2, 1, 0])
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

    if isinstance(frame, cudf.DataFrame):
        if hasattr(cond, "__cuda_array_interface__"):
            cond = cudf.DataFrame(
                cond, columns=frame._data.names, index=frame.index
            )
        elif not isinstance(cond, cudf.DataFrame):
            cond = frame.from_pandas(pd.DataFrame(cond))

        common_cols = set(frame._data.names).intersection(
            set(cond._data.names)
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
            cond.columns = frame.columns

        (source_df, others,) = _normalize_columns_and_scalars_type(
            frame, other
        )
        if isinstance(other, cudf.core.frame.Frame):
            others = others._data.columns

        out_df = cudf.DataFrame(index=frame.index)
        if len(frame._columns) != len(others):
            raise ValueError(
                """Replacement list length or number of dataframe columns
                should be equal to Number of columns of dataframe"""
            )
        for i, column_name in enumerate(frame._data.names):
            input_col = source_df._data[column_name]
            other_column = others[i]
            if column_name in cond._data:
                if isinstance(input_col, cudf.core.column.CategoricalColumn):
                    if cudf.utils.dtypes.is_scalar(other_column):
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
                        codes=cudf.core.column.as_column(
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
        if isinstance(other, cudf.DataFrame):
            raise NotImplementedError(
                "cannot align with a higher dimensional Frame"
            )
        input_col = frame._data[frame.name]
        cond = cudf.core.column.as_column(cond)
        if len(cond) != len(frame):
            raise ValueError(
                """Array conditional must be same shape as self"""
            )
        if cond.all():
            result = input_col
        else:
            (input_col, other,) = _normalize_columns_and_scalars_type(
                frame, other, inplace
            )

            if isinstance(input_col, cudf.core.column.CategoricalColumn):
                if cudf.utils.dtypes.is_scalar(other):
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

            if cudf.utils.dtypes.is_categorical_dtype(frame.dtype):
                result = cudf.core.column.build_categorical_column(
                    categories=frame._data[frame.name].categories,
                    codes=cudf.core.column.as_column(
                        result.base_data, dtype=result.dtype
                    ),
                    mask=result.base_mask,
                    size=result.size,
                    offset=result.offset,
                    ordered=frame._data[frame.name].ordered,
                )

        if isinstance(frame, cudf.Index):
            result = cudf.Index(result, name=frame.name)
        else:
            result = frame._copy_construct(data=result)

        return frame._mimic_inplace(result, inplace=inplace)
