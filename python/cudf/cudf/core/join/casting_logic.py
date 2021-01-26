# Copyright (c) 2021, NVIDIA CORPORATION.

from cudf.core.dtypes import CategoricalDtype
import pandas as pd
import cudf
import numpy as np
import warnings

def _input_to_libcudf_castrules_both_cat(lcol, rcol, how):
    '''
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



    '''
    ltype = lcol.dtype
    rtype = rcol.dtype

    # this function is only to be used to resolve the result when both
    # sides are categorical 
    if not isinstance(ltype, CategoricalDtype) and isinstance(rtype, CategoricalDtype):
        raise TypeError("Both operands must be CategoricalDtype")

    # true for every configuration 
    if ltype == rtype:
        return ltype

    elif how == 'inner':
        # two ways to fail:
        # 1. Equal categories, unequal ordering
        # 2. Unequal categories, either ordered
        if len(ltype.categories) == len(rtype.categories) and ltype.categories == rtype.categories: 
            if ltype.ordered != rtype.ordered:
                raise TypeError(
                    "Inner merge between an ordered and an"
                    "unordered categorical variable is ambiguous."
                )
        else:
            if ltype.ordered or rtype.ordered:
                raise TypeError(
                    "Inner merge between categoricals with "
                    "different categories is only valid when "
                    "neither side is ordered"
                )

        if not (ltype.ordered or rtype.ordered):
            # neiter ordered, so categories must be different
            return _input_to_libcudf_casting_rules_any(
                ltype.categories,
                rtype.categories,
                how)
        else:
            # only one of the operands is ordered
            raise TypeError(
                "inner merging on categorical variables when" \
                "only one side is ordered is ambiguous"
            )

    elif how in {'left', 'right'}:
        if how == 'left':
            major, minor = ltype, rtype
        elif how == 'right':
            major, minor = rtype, ltype
        
        # preserve ordering from the major table
        if minor.ordered and not major.ordered:
            warnings.warn(
                f"{how} join does not preserve ordering from"
                " the minor operand"
            )
            if major.categories.equals(minor.categories):
                return major
            else:
                raise TypeError(
                    f"{how} join when {how} table is unordered"
                    " and the other is not may only proceed when"
                    " categories are identical"
                )
        elif major.ordered and not minor.ordered:
            return major
        elif major.ordered and minor.ordered:
            # categories must be different
            raise TypeError(
                f"{how} join when the {how} categorical"
                " is unordered, but the other is not, is"
                " ambiguous"
            )
        else:
            # neither ordered, categories different
            return major

    elif how == 'outer':
        if (ltype.ordered != rtype.ordered) or (ltype.ordered and rtype.ordered):
            # One or both ordered
            raise TypeError(
                "Outer join may only proceed when neither"
                " categorical variable is ordered"
            )
        else:
            # neither ordered, so categories must be different
            new_cats = cudf.concat([ltype.categories, rtype.categories]).unique()
            return cudf.CategoricalDtype(categories=new_cats, ordered=False)


def _input_to_libcudf_castrules_one_cat(lcol, rcol, how):
    return 


def _input_to_libcudf_castrules_any_cat(lcol, rcol, how):

    l_is_cat = isinstance(lcol.dtype, CategoricalDtype)
    r_is_cat = isinstance(rcol.dtype, CategoricalDtype)

    if l_is_cat and r_is_cat:
        return _input_to_libcudf_castrules_both_cat(lcol, rcol, how)
    elif l_is_cat or r_is_cat:
        return _input_to_libcudf_castrules_one_cat(lcol, rcol, how)
    else:
        raise ValueError("Neither operand is categorical")


def _input_to_libcudf_casting_rules_any(lcol, rcol, how):
    """
    Determine what dtype the left and right hand
    input columns must be cast to for a libcudf
    join to proceed.
    """

    cast_warn = (
        "can't safely cast column from {} with type"
        " {} to {}, upcasting to {}"
    )
    ctgry_err = (
        "can't implicitly cast column {0} to categories"
        " from {1} during {1} join"
    )

    dtype_l = lcol.dtype
    dtype_r = rcol.dtype

    # if either side is categorical, different logic
    if isinstance(dtype_l, CategoricalDtype) or isinstance(dtype_r, CategoricalDtype):
        return _input_to_libcudf_castrules_any_cat(lcol, rcol, how)

    libcudf_join_type = None
    if pd.api.types.is_dtype_equal(dtype_l, dtype_r):
        libcudf_join_type = dtype_l
    elif how == "left":
        check_col = rcol.fillna(0)
        if not check_col.can_cast_safely(dtype_l):
            libcudf_join_type = _input_to_libcudf_casting_rules_any(
                lcol, rcol, "inner"
            )
            warnings.warn(
                cast_warn.format(
                    "right", dtype_r, dtype_l, libcudf_join_type
                )
            )
        else:
            libcudf_join_type = dtype_l
    elif how == "right":
        check_col = lcol.fillna(0)
        if not check_col.can_cast_safely(dtype_r):
            libcudf_join_type = input_to_libcudf_casting_rules_any(
                lcol, rcol, "inner"
            )
            warnings.warn(
                cast_warn.format(
                    "left", dtype_l, dtype_r, libcudf_join_type
                )
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
                libcudf_join_type = np.find_common_type(
                    [], [dtype_l, dtype_r]
                )
        elif np.issubdtype(dtype_l, np.datetime64) and np.issubdtype(
            dtype_r, np.datetime64
        ):
            libcudf_join_type = max(dtype_l, dtype_r)
    return libcudf_join_type
