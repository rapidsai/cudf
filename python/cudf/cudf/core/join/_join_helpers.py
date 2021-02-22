# Copyright (c) 2021, NVIDIA CORPORATION.

import warnings

import numpy as np
import pandas as pd

import cudf
from cudf.core.dtypes import CategoricalDtype


class _Indexer:
    # Indexer into a column (either a data column or index level).
    #
    # >>> df
    #    a
    # b
    # 4  1
    # 5  2
    # 6  3
    # >>> _Indexer("a", column=True).value(df)  # returns column "a" of df
    # >>> _Indexer("b", index=True).value(df)  # returns index level "b" of df

    def __init__(self, name, column=False, index=False):
        self.name = name
        self.column, self.index = column, index

    def value(self, obj):
        # get the column from `obj`
        if self.column:
            return obj._data[self.name]
        else:
            return obj._index._data[self.name]

    def set_value(self, obj, value):
        # set the colum in `obj`
        if self.column:
            obj._data[self.name] = value
        else:
            obj._index._data[self.name] = value

    def get_numeric_index(self, obj):
        # get the position of the column in `obj`
        # (counting any index columns)
        if self.column:
            index_nlevels = obj.index.nlevels if obj._index is not None else 0
            return index_nlevels + tuple(obj._data).index(self.name)
        else:
            return obj.index.names.index(self.name)


def _match_join_keys(lcol, rcol, how):
    # cast the keys lcol and rcol to a common dtype

    ltype = lcol.dtype
    rtype = rcol.dtype

    # if either side is categorical, different logic
    if isinstance(ltype, CategoricalDtype) or isinstance(
        rtype, CategoricalDtype
    ):
        return _match_join_categorical_keys(lcol, rcol, how)

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


def _match_join_categorical_keys(lcol, rcol, how):
    # cast the keys lcol and rcol to a common dtype
    # when at least one of them is a categorical type

    l_is_cat = isinstance(lcol.dtype, CategoricalDtype)
    r_is_cat = isinstance(rcol.dtype, CategoricalDtype)

    if l_is_cat and r_is_cat:
        # if both are categoricals, logic is complicated:
        return _match_join_categorical_keys_both(lcol, rcol, how)
    elif l_is_cat or r_is_cat:
        if l_is_cat and how in {"left", "leftsemi", "leftanti"}:
            return lcol.dtype
        common_type = (
            lcol.dtype.categories.dtype
            if l_is_cat
            else rcol.dtype.categories.dtype
        )
        return common_type
    else:
        raise ValueError("Neither operand is categorical")


def _match_join_categorical_keys_both(lcol, rcol, how):
    # cast lcol and rcol to a common type when they are *both*
    # categorical types.
    #
    # The commontype depends on both `how` and the specifics of the
    # categorical variables to be merged.

    ltype, rtype = lcol.dtype, rcol.dtype

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
        return _match_join_keys(ltype.categories, rtype.categories, how)
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
