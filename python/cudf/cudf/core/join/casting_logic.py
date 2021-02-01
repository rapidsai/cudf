# Copyright (c) 2021, NVIDIA CORPORATION.

import warnings

import numpy as np
import pandas as pd

import cudf
from cudf.core.dtypes import CategoricalDtype


def _input_to_libcudf_castrules_both_cat(lcol, rcol, how):
    """
    Based off the left and right operands, determine the libcudf
    merge dtype or error for corner cases where the merge cannot
    proceed. This function handles categorical variables.
    Categorical variable typecasting logic depends on both `how`
    and the specifics of the categorical variables to be merged.
    Merging categorical variables when only one side is ordered
    is ambiguous and not allowed. Merging when both categoricals
    are ordered is allowed, but only when the categories are
    exactly equal and have equal ordering, and will result in the
    common dtype.
    When both sides are unordered, the result categorical depends
    on the kind of join:
    - For inner joins, the result will be the intersection of the
    categories
    - For left or right joins, the result will be the the left or
    right dtype respectively. This extends to semi and anti joins.
    - For outer joins, the result will be the union of categories
    from both sides.

    """
    ltype = lcol.dtype
    rtype = rcol.dtype

    # this function is only to be used to resolve the result when both
    # sides are categorical
    if not isinstance(ltype, CategoricalDtype) and isinstance(
        rtype, CategoricalDtype
    ):
        raise TypeError("Both operands must be CategoricalDtype")

    # true for every configuration
    if ltype == rtype:
        return ltype

    # raise for any join where ordering doesn't match
    if ltype.ordered != rtype.ordered:
        raise TypeError(
            "Merging on categorical variables with mismatched"
            " ordering is ambiguous"
        )
    elif ltype.ordered and rtype.ordered:
        # if we get to here, categories must be what causes the
        # dtype equality check to fail. And we can never merge
        # two ordered categoricals with different categories
        raise TypeError(
            f"{how} merge between categoricals with "
            "different categories is only valid when "
            "neither side is ordered"
        )

    elif how == "inner":
        # neither ordered, so categories must be different
        # demote to underlying types
        return _input_to_libcudf_castrules_any(
            ltype.categories, rtype.categories, how
        )

    elif how == "left":
        return ltype
    elif how == "right":
        return rtype

    elif how == "outer":
        new_cats = cudf.concat([ltype.categories, rtype.categories]).unique()
        return cudf.CategoricalDtype(categories=new_cats, ordered=False)


def _input_to_libcudf_castrules_any_cat(lcol, rcol, how):

    l_is_cat = isinstance(lcol.dtype, CategoricalDtype)
    r_is_cat = isinstance(rcol.dtype, CategoricalDtype)

    if l_is_cat and r_is_cat:
        return _input_to_libcudf_castrules_both_cat(lcol, rcol, how)
    elif l_is_cat or r_is_cat:
        if l_is_cat and how == "left":
            return lcol.dtype
        if r_is_cat and how == "right":
            return rcol.dtype
        return (
            lcol.dtype.categories.dtype
            if l_is_cat
            else rcol.dtype.categories.dtype
        )
    else:
        raise ValueError("Neither operand is categorical")


def _input_to_libcudf_castrules_any(lcol, rcol, how):
    """
    Determine what dtype the left and right hand
    input columns must be cast to for a libcudf
    join to proceed.
    """

    cast_warn = (
        "can't safely cast column from {} with type"
        " {} to {}, upcasting to {}"
    )

    ltype = lcol.dtype
    rtype = rcol.dtype

    # if either side is categorical, different logic
    if isinstance(ltype, CategoricalDtype) or isinstance(
        rtype, CategoricalDtype
    ):
        return _input_to_libcudf_castrules_any_cat(lcol, rcol, how)

    libcudf_join_type = None
    if pd.api.types.is_dtype_equal(ltype, rtype):
        libcudf_join_type = ltype
    elif how == "left":
        check_col = rcol.fillna(0)
        if not check_col.can_cast_safely(ltype):
            libcudf_join_type = _input_to_libcudf_castrules_any(
                lcol, rcol, "inner"
            )
            warnings.warn(
                cast_warn.format("right", rtype, ltype, libcudf_join_type)
            )
        else:
            libcudf_join_type = ltype
    elif how == "right":
        check_col = lcol.fillna(0)
        if not check_col.can_cast_safely(rtype):
            libcudf_join_type = _input_to_libcudf_castrules_any(
                lcol, rcol, "inner"
            )
            warnings.warn(
                cast_warn.format("left", ltype, rtype, libcudf_join_type)
            )
        else:
            libcudf_join_type = rtype
    elif how in {"inner", "outer"}:
        if (np.issubdtype(ltype, np.number)) and (
            np.issubdtype(rtype, np.number)
        ):
            if ltype.kind == rtype.kind:
                # both ints or both floats
                libcudf_join_type = max(ltype, rtype)
            else:
                libcudf_join_type = np.find_common_type([], [ltype, rtype])
        elif np.issubdtype(ltype, np.datetime64) and np.issubdtype(
            rtype, np.datetime64
        ):
            libcudf_join_type = max(ltype, rtype)
    return libcudf_join_type


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

    l_is_cat = isinstance(ltype, CategoricalDtype)
    r_is_cat = isinstance(rtype, CategoricalDtype)

    # we  currently only need to do this for categorical variables
    if how == "inner":
        if l_is_cat and r_is_cat:
            merge_return_type = "category"
    elif how == "left":
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
