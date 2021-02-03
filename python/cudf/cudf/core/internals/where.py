import warnings

import numpy as np
import pandas as pd

import cudf


def _normalize_scalars(col, other):
    """
    Try to normalizes scalar values as per col dtype
    """
    if (
        other is not None
        and (isinstance(other, float) and not np.isnan(other))
    ) and (col.dtype.type(other) != other):
        raise TypeError(
            f"Cannot safely cast non-equivalent "
            f"{type(other).__name__} to {col.dtype.name}"
        )

    return (
        col.dtype.type(other)
        if (
            other is not None
            and (isinstance(other, float) and not np.isnan(other))
        )
        else other
    )


def _check_and_cast_columns(source_col, other_col, inplace):
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


def _check_and_cast_columns_with_scalar(source_col, other_scalar, inplace):
    """
    Returns type-casted column `source_col` & scalar `other_scalar`
    based on `inplace` parameter.
    """
    if cudf.utils.dtypes.is_categorical_dtype(source_col.dtype):
        return source_col, other_scalar

    device_scalar = cudf.Scalar(
        _normalize_scalars(source_col, other_scalar),
        dtype=source_col.dtype if other_scalar is None else None,
    )

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
        if not np.can_cast(device_scalar, source_col.dtype):
            warnings.warn(
                f"Type-casting from {device_scalar.dtype} "
                f"to {source_col.dtype}, there could be potential data loss"
            )
        return source_col, device_scalar.astype(source_col.dtype)
    else:
        if pd.api.types.is_numeric_dtype(source_col.dtype) and np.can_cast(
            other_scalar, source_col.dtype
        ):
            common_dtype = source_col.dtype
        else:
            common_dtype = cudf.utils.dtypes.find_common_type(
                [source_col.dtype, np.min_scalar_type(other_scalar)]
            )

        source_col = source_col.astype(common_dtype)
        return source_col, cudf.Scalar(other_scalar, dtype=common_dtype)


def _normalize_columns_and_scalars_type(frame, other, inplace=False):
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
        source_df = frame.copy()
        other_df = other.copy()
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

            source_df = frame.copy()
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
