# Copyright (c) 2021, NVIDIA CORPORATION.
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd

import cudf
from cudf.core.dtypes import CategoricalDtype

if TYPE_CHECKING:
    from cudf._typing import Dtype
    from cudf.core.column import ColumnBase
    from cudf.core.frame import Frame


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

    def __init__(self, name: Any, column=False, index=False):
        if column and index:
            raise ValueError("Cannot specify both column and index")
        self.name = name
        self.column, self.index = column, index

    def get(self, obj: Frame) -> ColumnBase:
        # get the column from `obj`
        if self.column:
            return obj._data[self.name]
        else:
            if obj._index is not None:
                return obj._index._data[self.name]
        raise KeyError()

    def set(self, obj: Frame, value: ColumnBase):
        # set the colum in `obj`
        if self.column:
            obj._data[self.name] = value
        else:
            if obj._index is not None:
                obj._index._data[self.name] = value
            else:
                raise KeyError()


def _frame_select_by_indexers(
    frame: Frame, indexers: Iterable[_Indexer]
) -> Frame:
    # Select columns from the given `Frame` using `indexers`,
    # and return a new `Frame`.
    index_data = frame._data.__class__()
    data = frame._data.__class__()

    for idx in indexers:
        if idx.index:
            index_data[idx.name] = idx.get(frame)
        else:
            data[idx.name] = idx.get(frame)

    result_index = cudf.Index._from_data(index_data) if index_data else None
    result = cudf.core.frame.Frame(data=data, index=result_index)
    return result


def _match_join_keys(lcol: ColumnBase, rcol: ColumnBase, how: str) -> Dtype:
    # cast the keys lcol and rcol to a common dtype

    ltype = lcol.dtype
    rtype = rcol.dtype

    # if either side is categorical, different logic
    if isinstance(ltype, CategoricalDtype) or isinstance(
        rtype, CategoricalDtype
    ):
        return _match_categorical_dtypes(ltype, rtype, how)

    if pd.api.types.is_dtype_equal(ltype, rtype):
        return ltype

    if (np.issubdtype(ltype, np.number)) and (np.issubdtype(rtype, np.number)):
        common_type = (
            max(ltype, rtype)
            if ltype.kind == rtype.kind
            else np.find_common_type([], (ltype, rtype))
        )

    elif np.issubdtype(ltype, np.datetime64) and np.issubdtype(
        rtype, np.datetime64
    ):
        common_type = max(ltype, rtype)

    if how == "left":
        if rcol.fillna(0).can_cast_safely(ltype):
            return ltype
        else:
            warnings.warn(
                f"Can't safely cast column from {rtype} to {ltype}, "
                "upcasting to {common_type}."
            )

    if common_type:
        return common_type

    return None


def _match_categorical_dtypes(ltype: Dtype, rtype: Dtype, how: str) -> Dtype:
    # cast the keys lcol and rcol to a common dtype
    # when at least one of them is a categorical type

    if isinstance(ltype, CategoricalDtype) and isinstance(
        rtype, CategoricalDtype
    ):
        # if both are categoricals, logic is complicated:
        return _match_categorical_dtypes_both(ltype, rtype, how)

    if isinstance(ltype, CategoricalDtype):
        if how in {"left", "leftsemi", "leftanti"}:
            return ltype
        common_type = ltype.categories.dtype
    elif isinstance(rtype, CategoricalDtype):
        common_type = rtype.categories.dtype
    return common_type


def _match_categorical_dtypes_both(
    ltype: CategoricalDtype, rtype: CategoricalDtype, how: str
) -> Dtype:
    # The commontype depends on both `how` and the specifics of the
    # categorical variables to be merged.

    # when both are ordered and both have the same categories,
    # no casting required:
    if ltype == rtype:
        return ltype

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

    # the following should now always hold
    assert not ltype.ordered and not rtype.ordered

    if how == "inner":
        # cast to category types -- we must cast them back later
        return _match_join_keys(
            ltype.categories._values, rtype.categories._values, how
        )
    elif how in {"left", "leftanti", "leftsemi"}:
        # always cast to left type
        return ltype
    else:
        # merge categories
        merged_categories = cudf.concat(
            [ltype.categories, rtype.categories]
        ).unique()
        common_type = cudf.CategoricalDtype(
            categories=merged_categories, ordered=False
        )
        return common_type


def _coerce_to_tuple(obj):
    if hasattr(obj, "__iter__") and not isinstance(obj, str):
        return tuple(obj)
    else:
        return (obj,)


def _coerce_to_list(obj):
    return list(_coerce_to_tuple(obj))
