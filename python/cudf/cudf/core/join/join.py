# Copyright (c) 2020-2021, NVIDIA CORPORATION.
import itertools
import warnings

import numpy as np
import pandas as pd

import cudf
from cudf import _lib as libcudf
from cudf._lib.join import compute_result_col_names
from cudf.core.dtypes import CategoricalDtype


class Merge(object):
    def __init__(
        self,
        lhs,
        rhs,
        on,
        left_on,
        right_on,
        left_index,
        right_index,
        how,
        sort,
        lsuffix,
        rsuffix,
        method,
        indicator,
        suffixes,
    ):
        """
        Manage the merging of two Frames.

        Parameters
        ----------
        lhs : Series or DataFrame
            The left operand of the merge
        rhs : Series or DataFrame
            The right operand of the merge
        on : string or list like
            A set of key columns in the left and right operands
            elements must be common to both frames
        left_on : string or list like
            A set of key columns in the left operand. Must be
            specified with right_on or right_index concurrently
        right_on : string or list like
            A set of key columns in the right operand. Must be
            specified with left_on or left_index concurrently
        left_index : bool
            Boolean flag indicating the left index column or columns
            are to be used as join keys in order.
        right_index : bool
            Boolean flag indicating the right index column or coumns
            are to be used as join keys in order.
        how : string
            The type of join. Possible values are
            'inner', 'outer', 'left', 'leftsemi' and 'leftanti'
        sort : bool
            Boolean flag indicating if the output Frame is to be
            sorted on the output's join keys, in left to right order.
        lsuffix : string
            The suffix to be appended to left hand column names that
            are found to exist in the right frame, but are not specified
            as join keys themselves.
        rsuffix : string
            The suffix to be appended to right hand column names that
            are found to exist in the left frame, but are not specified
            as join keys themselves.
        suffixes : list like
            Left and right suffixes specified together, unpacked into lsuffix
            and rsuffix.
        """
        self.lhs = lhs
        self.rhs = rhs
        self.left_index = left_index
        self.right_index = right_index
        self.method = method
        self.sort = sort

        # check that the merge is valid

        self.validate_merge_cfg(
            lhs,
            rhs,
            on,
            left_on,
            right_on,
            left_index,
            right_index,
            how,
            lsuffix,
            rsuffix,
            suffixes,
        )
        self.how = how
        self.preprocess_merge_params(
            on, left_on, right_on, lsuffix, rsuffix, suffixes
        )

    def perform_merge(self):
        """
        Call libcudf to perform a merge between the operands. If
        necessary, cast the input key columns to compatible types.
        Potentially also cast the output back to categorical.
        """
        output_dtypes = self.compute_output_dtypes()
        self.typecast_input_to_libcudf()
        libcudf_result = libcudf.join.join(
            self.lhs,
            self.rhs,
            self.how,
            self.method,
            left_on=self.left_on,
            right_on=self.right_on,
            left_index=self.left_index,
            right_index=self.right_index,
        )
        result = self.out_class._from_table(libcudf_result)
        result = self.typecast_libcudf_to_output(result, output_dtypes)
        if isinstance(result, cudf.Index):
            return result
        else:
            return result[
                compute_result_col_names(self.lhs, self.rhs, self.how)
            ]

    def preprocess_merge_params(
        self, on, left_on, right_on, lsuffix, rsuffix, suffixes
    ):
        """
        Translate a valid configuration of user input parameters into
        the subset of input configurations handled by the cython layer.
        Apply suffixes to columns.
        """

        self.out_class = cudf.DataFrame
        if isinstance(self.lhs, cudf.MultiIndex) or isinstance(
            self.rhs, cudf.MultiIndex
        ):
            self.out_class = cudf.MultiIndex
        elif isinstance(self.lhs, cudf.Index):
            self.out_class = self.lhs.__class__

        if on:
            on = [on] if isinstance(on, str) else list(on)
            left_on = right_on = on
        else:
            if left_on:
                left_on = (
                    [left_on] if isinstance(left_on, str) else list(left_on)
                )
            if right_on:
                right_on = (
                    [right_on] if isinstance(right_on, str) else list(right_on)
                )

        same_named_columns = set(self.lhs._data.keys()) & set(
            self.rhs._data.keys()
        )
        if not (left_on or right_on) and not (
            self.left_index and self.right_index
        ):
            left_on = right_on = list(same_named_columns)

        no_suffix_cols = []
        if left_on and right_on:
            no_suffix_cols = [
                left_name
                for left_name, right_name in zip(left_on, right_on)
                if left_name == right_name and left_name in same_named_columns
            ]

        if suffixes:
            lsuffix, rsuffix = suffixes
        for name in same_named_columns:
            if name not in no_suffix_cols:
                self.lhs.rename(
                    {name: f"{name}{lsuffix}"}, inplace=True, axis=1
                )
                self.rhs.rename(
                    {name: f"{name}{rsuffix}"}, inplace=True, axis=1
                )
                if left_on and name in left_on:
                    left_on[left_on.index(name)] = f"{name}{lsuffix}"
                if right_on and name in right_on:
                    right_on[right_on.index(name)] = f"{name}{rsuffix}"

        self.left_on = left_on if left_on is not None else []
        self.right_on = right_on if right_on is not None else []
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix

    @staticmethod
    def validate_merge_cfg(
        lhs,
        rhs,
        on,
        left_on,
        right_on,
        left_index,
        right_index,
        how,
        lsuffix,
        rsuffix,
        suffixes,
    ):
        """
        Error for various invalid combinations of merge input parameters
        """

        # must actually support the requested merge type
        if how not in {"left", "inner", "outer", "leftanti", "leftsemi"}:
            raise NotImplementedError(f"{how} merge not supported yet")

        # Passing 'on' with 'left_on' or 'right_on' is ambiguous
        if on and (left_on or right_on):
            raise ValueError(
                'Can only pass argument "on" OR "left_on" '
                'and "right_on", not a combination of both.'
            )

        # Can't merge on unnamed Series
        if (isinstance(lhs, cudf.Series) and not lhs.name) or (
            isinstance(rhs, cudf.Series) and not rhs.name
        ):
            raise ValueError("Can not merge on unnamed Series")

        # Keys need to be in their corresponding operands
        if on:
            if isinstance(on, str):
                on_keys = [on]
            elif isinstance(on, tuple):
                on_keys = list(on)
            else:
                on_keys = on
            for key in on_keys:
                if not (key in lhs._data.keys() and key in rhs._data.keys()):
                    raise KeyError(f"on key {on} not in both operands")
        elif left_on and right_on:
            left_on_keys = (
                [left_on] if not isinstance(left_on, list) else left_on
            )
            right_on_keys = (
                [right_on] if not isinstance(right_on, list) else right_on
            )

            for key in left_on_keys:
                if key not in lhs._data.keys():
                    raise KeyError(f'Key "{key}" not in left operand')
            for key in right_on_keys:
                if key not in rhs._data.keys():
                    raise KeyError(f'Key "{key}" not in right operand')

        # Require same total number of columns to join on in both operands
        len_left_on = 0
        len_right_on = 0
        if left_on:
            len_left_on += (
                len(left_on) if pd.api.types.is_list_like(left_on) else 1
            )
        if right_on:
            len_right_on += (
                len(right_on) if pd.api.types.is_list_like(right_on) else 1
            )
        if not (len_left_on + left_index * lhs._num_indices) == (
            len_right_on + right_index * rhs._num_indices
        ):
            raise ValueError(
                "Merge operands must have same number of join key columns"
            )

        # If nothing specified, must have common cols to use implicitly
        same_named_columns = set(lhs._data.keys()) & set(rhs._data.keys())
        if (
            not (left_index or right_index)
            and not (left_on or right_on)
            and len(same_named_columns) == 0
        ):
            raise ValueError("No common columns to perform merge on")

        if suffixes:
            lsuffix, rsuffix = suffixes
        for name in same_named_columns:
            if name == left_on == right_on:
                continue
            elif left_on and right_on:
                if (name in left_on and name in right_on) and (
                    left_on.index(name) == right_on.index(name)
                ):
                    continue
            else:
                if not (lsuffix or rsuffix):
                    raise ValueError(
                        "there are overlapping columns but "
                        "lsuffix and rsuffix are not defined"
                    )

    def typecast_input_to_libcudf(self):
        """
        Check each pair of join keys in the left and right hand
        operands and apply casting rules to match their types
        before passing the result to libcudf.
        """
        lhs_keys, rhs_keys, lhs_cols, rhs_cols = [], [], [], []
        if self.left_index:
            lhs_keys.append(self.lhs.index._data.keys())
            lhs_cols.append(self.lhs.index)
        if self.right_index:
            rhs_keys.append(self.rhs.index._data.keys())
            rhs_cols.append(self.rhs.index)
        if self.left_on:
            lhs_keys.append(self.left_on)
            lhs_cols.append(self.lhs)
        if self.right_on:
            rhs_keys.append(self.right_on)
            rhs_cols.append(self.rhs)

        for l_key_grp, r_key_grp, l_col_grp, r_col_grp in zip(
            lhs_keys, rhs_keys, lhs_cols, rhs_cols
        ):
            for l_key, r_key in zip(l_key_grp, r_key_grp):
                to_dtype = self.input_to_libcudf_casting_rules(
                    l_col_grp._data[l_key], r_col_grp._data[r_key], self.how
                )
                l_col_grp._data[l_key] = l_col_grp._data[l_key].astype(
                    to_dtype
                )
                r_col_grp._data[r_key] = r_col_grp._data[r_key].astype(
                    to_dtype
                )

    def _input_to_libcudf_castrules_any_cat(self, lcol, rcol, how):

        l_is_cat = isinstance(lcol.dtype, CategoricalDtype)
        r_is_cat = isinstance(rcol.dtype, CategoricalDtype)

        if l_is_cat and r_is_cat:
            return self._input_to_libcudf_castrules_both_cat(lcol, rcol, how)
        elif l_is_cat or r_is_cat:
            return self._input_to_libcudf_castrules_one_cat(lcol, rcol, how)
        else:
            raise ValueError("Neither operand is categorical")


    def _input_to_libcudf_castrules_both_cat(self, lcol, rcol, how):
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
            if ltype.categories == rtype.categories: 
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
                return self.input_to_libcudf_casting_rules(
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
            

                

    def _input_to_libcudf_castrules_one_cat(self, lcol, rcol, how):
        return 

    def input_to_libcudf_casting_rules(self, lcol, rcol, how):
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
            return self._input_to_libcudf_castrules_any_cat(lcol, rcol, how)

        libcudf_join_type = None
        if pd.api.types.is_dtype_equal(dtype_l, dtype_r):
            libcudf_join_type = dtype_l
        elif how == "left":
            check_col = rcol.fillna(0)
            if not check_col.can_cast_safely(dtype_l):
                libcudf_join_type = self.input_to_libcudf_casting_rules(
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
                libcudf_join_type = self.input_to_libcudf_casting_rules(
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

    def libcudf_to_output_casting_rules(self, lcol, rcol, how):
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
            if how == 'left':
                return dtype_l
            if how == 'right':
                return dtype_r
            elif how == 'outer':
                new_cats = cudf.concat([dtype_l.categories, dtype_r.categories]).unique()
                return cudf.CategoricalDtype(categories=new_cats, ordered=False)
            else:
                merge_return_type = "category"
        return merge_return_type

    def compute_output_dtypes(self):
        """
        Determine what datatypes should be applied to the result
        of a libcudf join, baesd on the original left and right
        frames.
        """

        index_dtypes = {}
        l_data_join_cols = {}
        r_data_join_cols = {}

        data_dtypes = {
            name: col.dtype
            for name, col in itertools.chain(
                self.lhs._data.items(), self.rhs._data.items()
            )
        }

        if self.left_index and self.right_index:
            l_idx_join_cols = list(self.lhs.index._data.values())
            r_idx_join_cols = list(self.rhs.index._data.values())
        elif self.left_on and self.right_index:
            # Keep the orignal dtypes in the LEFT index if possible
            # should trigger a bunch of no-ops
            l_idx_join_cols = list(self.lhs.index._data.values())
            r_idx_join_cols = list(self.lhs.index._data.values())
            for i, name in enumerate(self.left_on):
                l_data_join_cols[name] = self.lhs._data[name]
                r_data_join_cols[name] = list(self.rhs.index._data.values())[i]

        elif self.left_index and self.right_on:
            # see above
            l_idx_join_cols = list(self.rhs.index._data.values())
            r_idx_join_cols = list(self.rhs.index._data.values())
            for i, name in enumerate(self.right_on):
                l_data_join_cols[name] = list(self.lhs.index._data.values())[i]
                r_data_join_cols[name] = self.rhs._data[name]

        if self.left_on and self.right_on:
            l_data_join_cols = self.lhs._data
            r_data_join_cols = self.rhs._data

        if self.left_index or self.right_index:
            for i in range(len(self.lhs.index._data.items())):
                index_dtypes[i] = self.libcudf_to_output_casting_rules(
                    l_idx_join_cols[i], r_idx_join_cols[i], self.how
                )

        for name in itertools.chain(self.left_on, self.right_on):
            if name in self.left_on and name in self.right_on:
                data_dtypes[name] = self.libcudf_to_output_casting_rules(
                    l_data_join_cols[name], r_data_join_cols[name], self.how
                )
        return (index_dtypes, data_dtypes)

    def typecast_libcudf_to_output(self, output, output_dtypes):
        """
        Apply precomputed output index and data column data types
        to the output of a libcudf join.
        """

        index_dtypes, data_dtypes = output_dtypes
        if output._index and len(index_dtypes) > 0:
            for index_dtype, index_col_lbl, index_col in zip(
                index_dtypes.values(),
                output._index._data.keys(),
                output._index._data.values(),
            ):
                if index_dtype:
                    output._index._data[
                        index_col_lbl
                    ] = self._build_output_col(index_col, index_dtype)
            # reconstruct the Index object as the underlying data types
            # have changed:
            output._index = cudf.core.index.Index._from_table(output._index)

        for data_col_lbl, data_col in output._data.items():
            data_dtype = data_dtypes[data_col_lbl]
            if data_dtype:
                output._data[data_col_lbl] = self._build_output_col(
                    data_col, data_dtype
                )
        return output

    def _build_output_col(self, col, dtype):
        # problem:
        # equal dtypes, merge performed in int8 land via codes
        # then build_categorical_column works with those codes
        # unequal dtypes, merge performed in resolved dtype between both categories
        # now the resulting data is not codes indexed into categories
        if isinstance(
            dtype, (cudf.core.dtypes.CategoricalDtype, pd.CategoricalDtype)
        ):
            outcol = cudf.core.column.build_categorical_column(
                categories=dtype.categories,
                codes=col.set_mask(None),
                mask=col.base_mask,
                ordered=dtype.ordered,
            )
        else:
            outcol = col.astype(dtype)
        return outcol
