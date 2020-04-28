# Copyright (c) 2020, NVIDIA CORPORATION.

import warnings

import numpy as np
import pandas as pd
import itertools

import cudf._lib as libcudf
from cudf.utils.dtypes import is_categorical_dtype, is_datetime_dtype
import cudf

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
        )
        self.how = how
        self.preprocess_merge_params(on, left_on, right_on, lsuffix, rsuffix)

    def perform_merge(self):
        output_dtypes = self.compute_output_dtypes()
        self.typecast_input_to_libcudf()
        libcudf_result = libcudf.join.join(
            self.lhs,
            self.rhs,
            self.left_on,
            self.right_on,
            self.how,
            self.method,
            left_index=self.left_index,
            right_index=self.right_index,
        )
        result = self.typecast_libcudf_to_output(libcudf_result, output_dtypes)
        return result

    def preprocess_merge_params(self, on, left_on, right_on, lsuffix, rsuffix):
        if on:
            on = [on] if isinstance(on, str) else list(on)
            left_on = right_on = on
        else:
            on = []
        if left_on:
            left_on = [left_on] if isinstance(left_on, str) else list(left_on)
        else:
            left_on = []
        if right_on:
            right_on = (
                [right_on] if isinstance(right_on, str) else list(right_on)
            )
        else:
            right_on = []

        same_named_columns = set(self.lhs._data.keys()) & set(
            self.rhs._data.keys()
        )
        if not (left_on or right_on) and not (
            self.left_index and self.right_index
        ):
            left_on = right_on = list(same_named_columns)

        no_suffix_cols = []
        for name in same_named_columns:
            if left_on is not None and right_on is not None:
                if name in left_on and name in right_on:
                    if left_on.index(name) == right_on.index(name):
                        no_suffix_cols.append(name)

        for name in same_named_columns:
            if name not in no_suffix_cols:
                self.lhs.rename({name: f"{name}{lsuffix}"}, inplace=True)
                self.rhs.rename({name: f"{name}{rsuffix}"}, inplace=True)
                if name in left_on:
                    left_on[left_on.index(name)] = "%s%s" % (name, lsuffix)
                if name in right_on:
                    right_on[right_on.index(name)] = "%s%s" % (name, rsuffix)

        self.left_on = left_on if left_on is not None else []
        self.right_on = right_on if left_on is not None else []
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
    ):
        """
        Error for various combinations of merge input parameters
        """

        # must actually support the requested merge type
        if how not in ["left", "inner", "outer", "leftanti", "leftsemi"]:
            raise NotImplementedError(
                "{!r} merge not supported yet".format(how)
            )

        # Passing 'on' with 'left_on' or 'right_on' is ambiguous
        if on is not None:
            if left_on is not None or right_on is not None:
                raise ValueError(
                    'Can only pass argument "on" OR "left_on" '
                    'and "right_on", not a combination of both.'
                )
            # Keys in 'on' need to be in both operands
            on_keys = [on] if not isinstance(on, list) else on
            for key in on_keys:
                if not (key in lhs._data.keys() and key in rhs._data.keys()):
                    raise KeyError("on key {} not in both operands".format(on))

        # in this case, keys must be present in their corresponding operands
        if left_on is not None and right_on is not None:
            left_on_keys = (
                [left_on] if not isinstance(left_on, list) else left_on
            )
            right_on_keys = (
                [right_on] if not isinstance(right_on, list) else right_on
            )

            for key in left_on_keys:
                if key not in lhs._data.keys():
                    raise KeyError('Key "{}" not in left operand'.format(key))
            for key in right_on_keys:
                if key not in rhs._data.keys():
                    raise KeyError('Key "{}" not in right operand'.format(key))

        # Require same total number of columns to join on in both operands
        if left_on is None:
            len_left_on = 0
        elif not isinstance(left_on, list):
            len_left_on = 1
        else:
            len_left_on = len(left_on)

        if right_on is None:
            len_right_on = 0
        elif not isinstance(right_on, list):
            len_right_on = 1
        else:
            len_right_on = len(right_on)
        if not (len_left_on + left_index * lhs._num_indices) == (
            len_right_on + right_index * rhs._num_indices
        ):
            raise ValueError(
                "Merge operands must have same number of join key columns"
            )

        # If nothing specified, must have common cols to use implicitly
        same_named_columns = set(lhs._data.keys()) & set(rhs._data.keys())
        if not (left_index or right_index):
            if not (left_on or right_on):
                if len(same_named_columns) == 0:
                    raise ValueError("No common columns to perform merge on")

        if left_on is None:
            left_on = []
        if right_on is None:
            right_on = []
        for name in same_named_columns:
            if not (
                name in left_on
                and name in right_on
                and (left_on.index(name) == right_on.index(name))
            ):
                if not (lsuffix or rsuffix):
                    raise ValueError(
                        "there are overlapping columns but "
                        "lsuffix and rsuffix are not defined"
                    )

    def typecast_input_to_libcudf(self):
        if self.left_index and self.right_index:
            for ((kl, vl), (kr, vr)) in zip(
                self.lhs.index._data.items(), self.rhs.index._data.items()
            ):
                to_dtype = self.input_to_libcudf_casting_rules(
                    vl, vr, self.how
                )
                self.lhs.index._data[kl] = vl.astype(to_dtype)
                self.rhs.index._data[kr] = vr.astype(to_dtype)
            if self.left_on and self.right_on:
                for lcol, rcol in zip(self.left_on, self.right_on):
                    to_dtype = self.input_to_libcudf_casting_rules(
                        self.lhs._data[lcol], self.rhs._data[rcol], self.how
                    )
                    self.lhs._data[lcol] = self.lhs._data[lcol].astype(
                        to_dtype
                    )
                    self.rhs._data[rcol] = self.rhs._data[rcol].astype(
                        to_dtype
                    )
        elif self.left_index and self.right_on:
            for ((kl, vl), (kr, vr)) in zip(
                self.lhs.index._data.items(), self.rhs._data.items()
            ):
                to_dtype = self.input_to_libcudf_casting_rules(
                    vl, vr, self.how
                )
                self.lhs.index._data[kl] = vl.astype(to_dtype)
                self.rhs._data[kr] = vr.astype(to_dtype)
        elif self.right_index and self.left_on:
            for ((kl, vl), (kr, vr)) in zip(
                self.lhs._data.items(), self.rhs.index._data.items()
            ):
                to_dtype = self.input_to_libcudf_casting_rules(
                    vl, vr, self.how
                )
                self.lhs._data[kl] = vl.astype(to_dtype)
                self.rhs.index._data[kr] = vr.astype(to_dtype)

        elif self.left_on and self.right_on:
            for ((kl, vl), (kr, vr)) in zip(
                self.lhs._data.items(), self.rhs._data.items()
            ):
                to_dtype = self.input_to_libcudf_casting_rules(
                    vl, vr, self.how
                )
                self.lhs._data[kl] = vl.astype(to_dtype)
                self.rhs._data[kr] = vr.astype(to_dtype)

    def input_to_libcudf_casting_rules(self, lcol, rcol, how):
        cast_warn = "can't safely cast column {} from {} with type \
                        {} to {}, upcasting to {}"
        ctgry_err = "can't implicitly cast column {0} to categories \
                        from {1} during {1} join"

        dtype_l = lcol.dtype
        dtype_r = rcol.dtype
        libcudf_join_type = None
        if pd.api.types.is_dtype_equal(dtype_l, dtype_r):
            # if categorical and equal, children passed to libcudf
            libcudf_join_type = dtype_l
        elif is_categorical_dtype(dtype_l) and is_categorical_dtype(dtype_r):
            if how in ["inner", "outer"]:
                # join as the underlying category dtype winner
                libcudf_join_type = self.input_to_libcudf_casting_rules(
                    dtype_l.categories, dtype_r.categories, "inner"
                )
            elif how == "left":
                libcudf_join_type = self.input_to_libcudf_casting_rules(
                    dtype_l.categories, dtype_r.categories, "left"
                )

        elif how == "left":
            check_col = rcol.fillna(0)
            if not check_col.can_cast_safely(dtype_l):
                libcudf_join_type = self.input_to_libcudf_casting_rules(
                    lcol, rcol, "inner"
                )
                warnings.warn(
                    cast_warn.format(
                        rcol, "right", dtype_r, dtype_l, libcudf_join_type
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
                        lcol, "left", dtype_l, dtype_r, libcudf_join_type
                    )
                )
            else:
                libcudf_join_type = dtype_r

        elif is_categorical_dtype(dtype_l):
            if how == "right":
                raise ValueError(ctgry_err.format(rcol, "right"))
            libcudf_join_type = lcol.cat().categories.dtype
        elif is_categorical_dtype(dtype_r):
            if how == "left":
                raise ValueError(ctgry_err.format(lcol, "left"))
            libcudf_join_type = rcol.cat().categories.dtype
        elif how in ["inner", "outer"]:
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
            elif is_datetime_dtype(dtype_l) and is_datetime_dtype(dtype_r):
                libcudf_join_type = max(dtype_l, dtype_r)
        return libcudf_join_type

    def libcudf_to_output_casting_rules(self, lcol, rcol, how):

        dtype_l = lcol.dtype
        dtype_r = rcol.dtype

        merge_return_type = None
        # we  currently only need to do this for categorical variables
        if (is_categorical_dtype(dtype_l) or is_categorical_dtype(dtype_r)):
            if how in ['inner', 'left']:
                merge_return_type = dtype_l
            elif how == 'outer':
                new_cats = cudf.concat(dtype_l.categories, dtype_r.categories)
                merge_return_type = cudf.core.dtypes.CategoricalDtype(categories=new_cats)
        return merge_return_type

    def compute_output_dtypes(self):
        index_dtypes = []
        data_dtypes = {}
        for name, col in itertools.chain(self.lhs._data.items(), self.rhs._data.items()):
                data_dtypes[name] = col.dtype

        if self.left_index and self.right_index:
            for ((kl, vl), (kr, vr)) in zip(
                self.lhs.index._data.items(),
                self.rhs.index._data.items()
            ):
                to_dtype = self.libcudf_to_output_casting_rules(vl, vr, self.how)
                index_dtypes.append(to_dtype)
            if self.left_on and self.right_on:
                for ((kl, vl), (kr, vr)) in zip(
                    self.lhs[self.left_on]._data.items(),
                    self.rhs[self.right_on]._data.items()
                ):
                    to_dtype = self.libcudf_to_output_casting_rules(vl, vr, self.how)
                    data_dtypes[kl] = to_dtype
                    data_dtypes[kr] = to_dtype
        elif self.left_index and self.right_on:
            for ((kl, vl), (kr, vr)) in zip(
                self.lhs.index._data.items(),
                self.rhs[self.right_on]._data.items()
            ):
                to_dtype = self.libcudf_to_output_casting_rules(vl, vr, self.how)
                data_dtypes[kr] = to_dtype
                index_dtypes = [col.dtype for col in self.rhs.index._data.values()]
        elif self.right_index and self.left_on:
            for ((kl, vl), (kr, vr)) in zip(
                self.lhs[self.left_on]._data.items(),
                self.rhs.index._data.items()
            ):
                to_dtype = self.libcudf_to_output_casting_rules(vl, vr, self.how)
                data_dtypes[kl] = to_dtype
                index_dtypes = [col.dtype for col in self.lhs.index._data.values()]
        elif self.left_on and self.right_on:
            for ((kl, vl), (kr, vr)) in zip(
                self.lhs[self.left_on]._data.items(),
                self.rhs[self.right_on]._data.items()
            ):
                to_dtype = self.libcudf_to_output_casting_rules(vl, vr, self.how)
                data_dtypes[kl] = to_dtype
                data_dtypes[kr] = to_dtype
        return (index_dtypes, data_dtypes)

    def typecast_libcudf_to_output(self, output, output_dtypes):
        index_dtypes, data_dtypes = output_dtypes
        if output._index is not None:
            for i, (idx_col_lbl, index_col) in enumerate(output._index._data.items()):

                if index_dtypes[i] is not None:
                    output._index._data[idx_col_lbl] = index_col.astype(index_dtypes[i])
        for data_col_lbl, data_col in output._data.items():
            if data_dtypes[data_col_lbl] is not None:
                output._data[data_col_lbl] = data_col.astype(data_dtypes[data_col_lbl])
        return output


    @staticmethod
    def compute_result_col_names(lhs, rhs, how):
        if how in ("left", "inner", "outer"):
            # the result cols are all the left columns (incl. common ones)
            # + all the right columns (excluding the common ones)
            result_col_names = [None] * len(
                lhs._data.keys() | rhs._data.keys()
            )
            ix = 0
            for name in lhs._data.keys():
                result_col_names[ix] = name
                ix += 1
            for name in rhs._data.keys():
                if name not in lhs._data.keys():
                    nom = name
                    result_col_names[ix] = nom
                    ix += 1
        elif how in ("leftsemi", "leftanti"):
            # the result columns are just all the left columns
            result_col_names = list(lhs._data.keys())
        return result_col_names
