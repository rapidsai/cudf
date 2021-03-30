# Copyright (c) 2021, NVIDIA CORPORATION.
from __future__ import annotations

import collections
import warnings
from typing import TYPE_CHECKING, Any, Iterable, Tuple

import numpy as np
import pandas as pd

import cudf
from cudf.core.dtypes import CategoricalDtype

if TYPE_CHECKING:
    from cudf.core.column import CategoricalColumn, ColumnBase
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

    def set(self, obj: Frame, value: ColumnBase, validate=False):
        # set the colum in `obj`
        if self.column:
            obj._data.set_by_label(self.name, value, validate=validate)
        else:
            if obj._index is not None:
                obj._index._data.set_by_label(
                    self.name, value, validate=validate
                )
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
            index_data.set_by_label(idx.name, idx.get(frame), validate=False)
        else:
            data.set_by_label(idx.name, idx.get(frame), validate=False)

    result_index = cudf.Index._from_data(index_data) if index_data else None
    result = cudf.core.frame.Frame(data=data, index=result_index)
    return result


def _match_join_keys(
    lcol: ColumnBase, rcol: ColumnBase, how: str
) -> Tuple[ColumnBase, ColumnBase]:
    # returns the common dtype that lcol and rcol should be casted to,
    # before they can be used as left and right join keys.
    # If no casting is necessary, returns None

    common_type = None

    # cast the keys lcol and rcol to a common dtype
    ltype = lcol.dtype
    rtype = rcol.dtype

    # if either side is categorical, different logic
    if isinstance(ltype, CategoricalDtype) or isinstance(
        rtype, CategoricalDtype
    ):
        return _match_categorical_dtypes(lcol, rcol, how)

    if pd.api.types.is_dtype_equal(ltype, rtype):
        return lcol, rcol

    if isinstance(ltype, cudf.Decimal64Dtype) or isinstance(
        rtype, cudf.Decimal64Dtype
    ):
        raise TypeError(
            "Decimal columns can only be merged with decimal columns "
            "of the same precision and scale"
        )

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
            return lcol, rcol.astype(ltype)
        else:
            warnings.warn(
                f"Can't safely cast column from {rtype} to {ltype}, "
                "upcasting to {common_type}."
            )

    return lcol.astype(common_type), rcol.astype(common_type)


def _match_categorical_dtypes(
    lcol: ColumnBase, rcol: ColumnBase, how: str
) -> Tuple[ColumnBase, ColumnBase]:
    # cast the keys lcol and rcol to a common dtype
    # when at least one of them is a categorical type
    ltype, rtype = lcol.dtype, rcol.dtype

    if isinstance(lcol, cudf.core.column.CategoricalColumn) and isinstance(
        rcol, cudf.core.column.CategoricalColumn
    ):
        # if both are categoricals, logic is complicated:
        return _match_categorical_dtypes_both(lcol, rcol, how)

    if isinstance(ltype, CategoricalDtype):
        if how in {"left", "leftsemi", "leftanti"}:
            return lcol, rcol.astype(ltype)
        common_type = ltype.categories.dtype
    elif isinstance(rtype, CategoricalDtype):
        common_type = rtype.categories.dtype
    return lcol.astype(common_type), rcol.astype(common_type)


def _match_categorical_dtypes_both(
    lcol: CategoricalColumn, rcol: CategoricalColumn, how: str
) -> Tuple[ColumnBase, ColumnBase]:
    # The commontype depends on both `how` and the specifics of the
    # categorical variables to be merged.

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

    # the following should now always hold
    assert not ltype.ordered and not rtype.ordered

    if how == "inner":
        # cast to category types -- we must cast them back later
        return _match_join_keys(
            lcol.cat()._decategorize(), rcol.cat()._decategorize(), how,
        )
    elif how in {"left", "leftanti", "leftsemi"}:
        # always cast to left type
        return lcol, rcol.astype(ltype)
    else:
        # merge categories
        merged_categories = cudf.concat(
            [ltype.categories, rtype.categories]
        ).unique()
        common_type = cudf.CategoricalDtype(
            categories=merged_categories, ordered=False
        )
        return lcol.astype(common_type), rcol.astype(common_type)


def _coerce_to_tuple(obj):
    if isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str):
        return tuple(obj)
    else:
        return (obj,)
