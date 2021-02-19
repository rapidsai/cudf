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


def _coerce_to_tuple(obj):
    if hasattr(obj, "__iter__") and not isinstance(obj, str):
        return tuple(obj)
    else:
        return (obj,)


def _coerce_to_list(obj):
    return list(_coerce_to_tuple(obj))


def _cast_join_categorical_keys_both(lcol, rcol, how):
    # cast lcol and rcol to a common type when they are *both*
    # categorical types.
    #
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
        # demote to underlying types -- we will promote them back later
        return _cast_join_keys(ltype.categories, rtype.categories, how)
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


def _cast_join_categorical_keys(lcol, rcol, how):
    # cast the keys lcol and rcol to a common dtype
    # when at least one of them is a categorical type

    l_is_cat = isinstance(lcol.dtype, CategoricalDtype)
    r_is_cat = isinstance(rcol.dtype, CategoricalDtype)

    if l_is_cat and r_is_cat:
        # if both are categoricals, logic is complicated:
        return _cast_join_categorical_keys_both(lcol, rcol, how)
    elif l_is_cat or r_is_cat:
        if l_is_cat and how in {"left", "leftsemi", "leftanti"}:
            return (lcol, rcol.astype(lcol.dtype))
        common_type = (
            lcol.dtype.categories.dtype
            if l_is_cat
            else rcol.dtype.categories.dtype
        )
        return lcol.astype(common_type), rcol.astype(common_type)
    else:
        raise ValueError("Neither operand is categorical")


def _cast_join_keys(lcol, rcol, how):
    # cast the keys lcol and rcol to a common dtype

    ltype = lcol.dtype
    rtype = rcol.dtype

    # if either side is categorical, different logic
    if isinstance(ltype, CategoricalDtype) or isinstance(
        rtype, CategoricalDtype
    ):
        return _cast_join_categorical_keys(lcol, rcol, how)

    if pd.api.types.is_dtype_equal(ltype, rtype):
        return lcol, rcol

    if (np.issubdtype(ltype, np.number)) and (np.issubdtype(rtype, np.number)):
        common_type = (
            max(ltype, type)
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
                "Can't safely cast column from {rtype} to {ltype}, "
                "upcasting to {common_type}."
            )

    if common_type:
        lcol, rcol = lcol.astype(common_type), rcol.astype(common_type)

    return lcol, rcol


def _libcudf_to_output_castrules(lcol, rcol, how):
    """
    Determine what dtype an output merge key column should be
    cast to after it has been processed by libcudf. Determine
    if a column should be promoted to a categorical datatype.
    For inner merges between unordered categoricals, we get a
    new categorical variable containing the intersection of
    the two source variables. For left or right joins, we get
    the original categorical variable from whichever was the
    major operand of the join, e.g. left for a left join or
    right for a right join. In the case of an outer join, the
    result will be a new categorical variable with both sets
    of categories.
    """
    merge_return_type = None

    ltype = lcol.dtype
    rtype = rcol.dtype

    if pd.api.types.is_dtype_equal(ltype, rtype):
        return ltype

    merge_return_type = _cast_join_keys(lcol, rcol, how)

    l_is_cat = isinstance(ltype, CategoricalDtype)
    r_is_cat = isinstance(rtype, CategoricalDtype)

    # we currently only need to do this for categorical variables
    if how == "inner":
        if l_is_cat and r_is_cat:
            merge_return_type = "category"
    elif how in {"left", "leftsemi", "leftanti"}:
        if l_is_cat:
            merge_return_type = ltype
    elif how == "right":
        if r_is_cat:
            merge_return_type = rtype
    elif how == "outer":
        if l_is_cat and r_is_cat:
            new_cats = cudf.concat(
                [ltype.categories, rtype.categories]
            ).unique()
            merge_return_type = cudf.CategoricalDtype(
                categories=new_cats, ordered=ltype.ordered
            )
    return merge_return_type
