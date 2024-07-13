# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from __future__ import annotations

import warnings
from collections import abc
from typing import TYPE_CHECKING, Any, cast

import numpy as np

import cudf
from cudf.api.types import is_decimal_dtype, is_dtype_equal, is_numeric_dtype
from cudf.core.column import CategoricalColumn
from cudf.core.dtypes import CategoricalDtype

if TYPE_CHECKING:
    from cudf.core.column import ColumnBase


class _Indexer:
    # Indexer into a column (either a data column or index level).
    #
    # >>> df
    #    a
    # b
    # 4  1
    # 5  2
    # 6  3
    # >>> _Indexer("a", column=True).get(df)  # returns column "a" of df
    # >>> _Indexer("b", index=True).get(df)  # returns index level "b" of df

    def __init__(self, name: Any):
        self.name = name


class _ColumnIndexer(_Indexer):
    def get(self, obj: cudf.DataFrame) -> ColumnBase:
        return obj._data[self.name]

    def set(self, obj: cudf.DataFrame, value: ColumnBase, validate=False):
        obj._data.set_by_label(self.name, value, validate=validate)


class _IndexIndexer(_Indexer):
    def get(self, obj: cudf.DataFrame) -> ColumnBase:
        return obj.index._data[self.name]

    def set(self, obj: cudf.DataFrame, value: ColumnBase, validate=False):
        obj.index._data.set_by_label(self.name, value, validate=validate)


def _match_join_keys(
    lcol: ColumnBase, rcol: ColumnBase, how: str
) -> tuple[ColumnBase, ColumnBase]:
    # Casts lcol and rcol to a common dtype for use as join keys. If no casting
    # is necessary, they are returned as is.

    common_type = None

    # cast the keys lcol and rcol to a common dtype
    ltype = lcol.dtype
    rtype = rcol.dtype

    # if either side is categorical, different logic
    left_is_categorical = isinstance(ltype, CategoricalDtype)
    right_is_categorical = isinstance(rtype, CategoricalDtype)
    if left_is_categorical and right_is_categorical:
        return _match_categorical_dtypes_both(
            cast(CategoricalColumn, lcol), cast(CategoricalColumn, rcol), how
        )
    elif left_is_categorical or right_is_categorical:
        if left_is_categorical:
            if how in {"left", "leftsemi", "leftanti"}:
                return lcol, rcol.astype(ltype)
            common_type = ltype.categories.dtype
        else:
            common_type = rtype.categories.dtype
        common_type = cudf.utils.dtypes._dtype_pandas_compatible(common_type)
        return lcol.astype(common_type), rcol.astype(common_type)

    if is_dtype_equal(ltype, rtype):
        return lcol, rcol

    if is_decimal_dtype(ltype) or is_decimal_dtype(rtype):
        raise TypeError(
            "Decimal columns can only be merged with decimal columns "
            "of the same precision and scale"
        )

    if (
        is_numeric_dtype(ltype)
        and is_numeric_dtype(rtype)
        and not (ltype.kind == "m" or rtype.kind == "m")
    ):
        common_type = (
            max(ltype, rtype)
            if ltype.kind == rtype.kind
            else np.result_type(ltype, rtype)
        )
    elif (ltype.kind == "M" and rtype.kind == "M") or (
        ltype.kind == "m" and rtype.kind == "m"
    ):
        common_type = max(ltype, rtype)
    elif ltype.kind in "mM" and not rcol.fillna(0).can_cast_safely(ltype):
        raise TypeError(
            f"Cannot join between {ltype} and {rtype}, please type-cast both "
            "columns to the same type."
        )
    elif rtype.kind in "mM" and not lcol.fillna(0).can_cast_safely(rtype):
        raise TypeError(
            f"Cannot join between {rtype} and {ltype}, please type-cast both "
            "columns to the same type."
        )

    if how == "left" and rcol.fillna(0).can_cast_safely(ltype):
        return lcol, rcol.astype(ltype)

    return lcol.astype(common_type), rcol.astype(common_type)


def _match_categorical_dtypes_both(
    lcol: CategoricalColumn, rcol: CategoricalColumn, how: str
) -> tuple[ColumnBase, ColumnBase]:
    ltype, rtype = lcol.dtype, rcol.dtype

    # when both are ordered and both have the same categories,
    # no casting required:
    if ltype == rtype:
        return lcol, rcol

    # Merging categorical variables when only one side is ordered is
    # ambiguous and not allowed.
    if ltype.ordered != rtype.ordered:
        raise TypeError(
            "Merging on categorical variables with mismatched"
            " ordering is ambiguous"
        )

    if ltype.ordered and rtype.ordered:
        # if we get to here, categories must be what causes the
        # dtype equality check to fail. And we can never merge
        # two ordered categoricals with different categories
        raise TypeError(
            f"{how} merge between categoricals with "
            "different categories is only valid when "
            "neither side is ordered"
        )

    if how == "inner":
        # cast to category types -- we must cast them back later
        return _match_join_keys(
            lcol._get_decategorized_column(),
            rcol._get_decategorized_column(),
            how,
        )
    elif how in {"left", "leftanti", "leftsemi"}:
        # always cast to left type
        return lcol, rcol.astype(ltype)
    else:
        # merge categories
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            merged_categories = cudf.concat(
                [ltype.categories, rtype.categories]
            ).unique()
        common_type = cudf.CategoricalDtype(
            categories=merged_categories, ordered=False
        )
        return lcol.astype(common_type), rcol.astype(common_type)


def _coerce_to_tuple(obj):
    if isinstance(obj, abc.Iterable) and not isinstance(obj, str):
        return tuple(obj)
    else:
        return (obj,)
