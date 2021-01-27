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
    For `inner` joins:
        - If the left and right dtypes have equal categories and
        equal ordering, the join will proceed and the libcudf
        cython code will automatically forward the op to children
        - If the the left and right dtypes have equal categories
        but unequal ordering, the join can not proceed.
        - If the left and right dtypes have unequal categories
        and neither are ordered, the underlying dtype of the
        categories from both sides are fed to the non-categorical
        casting rules, and a new unordered categorical is later
        generated from the factorized subset of keys in the result
        - If the left and right dtypes have unequal categories and
        either side is ordered, the output type is ambiguous and
        the join can not proceed.
    For `left`, `right`, `semi` and `anti` joins:
        - If the left and right dtypes have equal categories and
        equal ordering, the join will proceed as in `inner`.
        - If the left and right dtypes have equal categories and
        unequal ordering, the join will proceed and the ordering
        from the major operand will take precedence.
        - If the left and right dtypes have unequal categories and
        neither are ordered, the join will proceed, and the major
        operands dtype will be retained in the output.
        - If the left and right dtypes have unequal categories and
        only one is ordered, the join will proceed and the major
        operands ordering will take precedence.
        - If the left and right dtypes have unequal categories and
        both are ordered, the join can not proceed.
    for `outer` joins:
        - If the left and right dtypes have equal categories and
        equal ordering, the join will proceed as in `inner`.
        - If the left and right dtypes have equal categories and
        unequal ordering, the join can not proceed.
        - If the left and right dtypes have unequal categories
        and neither are ordered, the underlying dtype of the
        categories from both sides are fed to the non-categorical
        casting rules, and a new unordered categorical is later
        generated from the deduped union of both source categories
        - If the left and right dtypes have unequal categories and
        either one or both are ordered, the join can not proceed.



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

    elif how == "inner":
        if not (ltype.ordered or rtype.ordered):
            # neiter ordered, so categories must be different
            # demote to underlying types
            return _input_to_libcudf_castrules_any(
                ltype.categories, rtype.categories, how
            )
        else:
            raise TypeError(
                "Inner merge between categoricals with "
                "different categories is only valid when "
                "neither side is ordered"
            )

    elif how in {"left", "right"}:
        if not (ltype.ordered or rtype.ordered):
            if how == "left":
                return ltype
            elif how == "right":
                return rtype
        else:
            raise TypeError(
                f"{how} merge between categoricals with "
                "different categories is only valid when "
                "neither side is ordered"
            )


    elif how == "outer":
        if not (ltype.ordered or rtype.ordered):
            # neither ordered, so categories must be different
            new_cats = cudf.concat(
                [ltype.categories, rtype.categories]
            ).unique()
            return cudf.CategoricalDtype(categories=new_cats, ordered=False)
        else:
            raise TypeError(
                f"{how} merge between categoricals with "
                "different categories is only valid when "
                "neither side is ordered"
            )


def _input_to_libcudf_castrules_one_cat(lcol, rcol, how):
    l_is_cat = isinstance(lcol.dtype, CategoricalDtype)
    r_is_cat = isinstance(rcol.dtype, CategoricalDtype)

    if l_is_cat:
        return lcol.dtype.categories.dtype
    elif r_is_cat:
        return rcol.dtype.categories.dtype


def _input_to_libcudf_castrules_any_cat(lcol, rcol, how):

    l_is_cat = isinstance(lcol.dtype, CategoricalDtype)
    r_is_cat = isinstance(rcol.dtype, CategoricalDtype)

    if l_is_cat and r_is_cat:
        return _input_to_libcudf_castrules_both_cat(lcol, rcol, how)
    elif l_is_cat or r_is_cat:
        return _input_to_libcudf_castrules_one_cat(lcol, rcol, how)
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

    dtype_l = lcol.dtype
    dtype_r = rcol.dtype

    # if either side is categorical, different logic
    if isinstance(dtype_l, CategoricalDtype) or isinstance(
        dtype_r, CategoricalDtype
    ):
        return _input_to_libcudf_castrules_any_cat(lcol, rcol, how)

    libcudf_join_type = None
    if pd.api.types.is_dtype_equal(dtype_l, dtype_r):
        libcudf_join_type = dtype_l
    elif how == "left":
        check_col = rcol.fillna(0)
        if not check_col.can_cast_safely(dtype_l):
            libcudf_join_type = _input_to_libcudf_castrules_any(
                lcol, rcol, "inner"
            )
            warnings.warn(
                cast_warn.format("right", dtype_r, dtype_l, libcudf_join_type)
            )
        else:
            libcudf_join_type = dtype_l
    elif how == "right":
        check_col = lcol.fillna(0)
        if not check_col.can_cast_safely(dtype_r):
            libcudf_join_type = _input_to_libcudf_castrules_any(
                lcol, rcol, "inner"
            )
            warnings.warn(
                cast_warn.format("left", dtype_l, dtype_r, libcudf_join_type)
            )
        else:
            libcudf_join_type = dtype_r
    elif how in {"inner", "outer"}:
        if (np.issubdtype(dtype_l, np.number)) and (
            np.issubdtype(dtype_r, np.number)
        ):
            if dtype_l.kind == dtype_r.kind:
                # both ints or both floats
                libcudf_join_type = max(dtype_l, dtype_r)
            else:
                libcudf_join_type = np.find_common_type([], [dtype_l, dtype_r])
        elif np.issubdtype(dtype_l, np.datetime64) and np.issubdtype(
            dtype_r, np.datetime64
        ):
            libcudf_join_type = max(dtype_l, dtype_r)
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
    of categories. Ordering is retained when using left/right
    joins.
    """

    dtype_l = lcol.dtype
    dtype_r = rcol.dtype
    merge_return_type = None
    # we  currently only need to do this for categorical variables
    if isinstance(dtype_l, CategoricalDtype) and isinstance(
        dtype_r, CategoricalDtype
    ):
        if pd.api.types.is_dtype_equal(dtype_l, dtype_r):
            if how == "inner":
                return dtype_l
        if how == "left":
            return dtype_l
        if how == "right":
            return dtype_r
        elif how == "outer":
            new_cats = cudf.concat(
                [dtype_l.categories, dtype_r.categories]
            ).unique()
            return cudf.CategoricalDtype(categories=new_cats, ordered=False)
        else:
            merge_return_type = "category"
    return merge_return_type
