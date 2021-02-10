# Copyright (c) 2020-2021, NVIDIA CORPORATION.
from collections import OrderedDict, namedtuple

import cudf
from cudf import _lib as libcudf
from cudf.core.join.casting_logic import (
    _input_to_libcudf_castrules_any,
    _libcudf_to_output_castrules,
)


class _MISSING_TYPE:
    pass


MISSING = _MISSING_TYPE()


class ColumnView:
    # A ColumnView represents one column of a Series
    # or DataFrame - either an index column or a
    # data column

    # we need a different sentinel value than `None`
    # because `None` is totally a valid index/column name
    def __init__(self, obj, column=MISSING, index=MISSING):
        self.obj = obj
        self.column, self.index = column, index

    def get_numeric_index(self):
        # get the position of the column (including any index columns)
        if self.index is MISSING:
            index_nlevels = (
                self.obj.index.nlevels if self.obj._index is not None else 0
            )
            return index_nlevels + tuple(self.obj._data).index(self.column)
        else:
            return self.obj.index.names.index(self.index)

    @property
    def is_index_level(self):
        # True if this is an index column
        return self.index is not MISSING

    @property
    def name(self):
        # get the name of the column
        if self.index is MISSING:
            return self.column
        else:
            return self.index

    @property
    def value(self):
        # get the column
        if self.index is MISSING:
            return self.obj._data[self.name]
        else:
            return self.obj._index._data[self.name]

    def set_value(self, value):
        # set the colum
        if self.index is MISSING:
            self.obj._data[self.name] = value
        else:
            self.obj._index._data[self.name] = value


JoinKeys = namedtuple("JoinKeys", ["left", "right"])


def Merge(
    lhs,
    rhs,
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    how="inner",
    sort=False,
    lsuffix="_x",
    rsuffix="_y",
    method=None,
    indicator=None,
    suffixes=None,
):
    if how not in {"leftsemi", "leftanti"}:
        return MergeBase(
            lhs,
            rhs,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            sort=sort,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            method=method,
            indicator=indicator,
            suffixes=suffixes,
        )
    else:
        return MergeSemi(
            lhs,
            rhs,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            sort=sort,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            method=method,
            indicator=indicator,
            suffixes=suffixes,
        )


class MergeBase(object):
    def __init__(
        self,
        lhs,
        rhs,
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        how="inner",
        sort=False,
        lsuffix="_x",
        rsuffix="_y",
        method=None,
        indicator=None,
        suffixes=None,
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
        self.validate_merge_params(
            lhs,
            rhs,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            suffixes=suffixes,
        )

        # warning: self.lhs and self.rhs are mutated both before
        # and after the join
        self.lhs = lhs.copy(deep=False)
        self.rhs = rhs.copy(deep=False)

        self.on = on
        self.left_on = left_on
        self.right_on = right_on
        self.left_index = left_index
        self.right_index = right_index
        self.how = how
        self.sort = sort
        self.lsuffix = lsuffix
        self.rsuffix = rsuffix
        self.suffixes = suffixes

        self.out_class = cudf.DataFrame
        if isinstance(self.lhs, cudf.MultiIndex) or isinstance(
            self.rhs, cudf.MultiIndex
        ):
            self.out_class = cudf.MultiIndex
        elif isinstance(self.lhs, cudf.Index):
            self.out_class = self.lhs.__class__

        self.compute_join_keys()

    def compute_join_keys(self):

        if (
            self.left_index
            or self.right_index
            or self.left_on
            or self.right_on
        ):
            if self.left_index:
                left_keys = [
                    ColumnView(obj=self.lhs, index=on)
                    for on in self.lhs.index.names
                ]
            else:
                # TODO: require left_on or left_index to be specified
                left_keys = [
                    ColumnView(obj=self.lhs, column=on)
                    for on in _coerce_to_tuple(self.left_on)
                ]
            if self.right_index:
                right_keys = [
                    ColumnView(obj=self.rhs, index=on)
                    for on in self.rhs.index.names
                ]
            else:
                # TODO: require right_on or right_index to be specified
                right_keys = [
                    ColumnView(obj=self.rhs, column=on)
                    for on in _coerce_to_tuple(self.right_on)
                ]
        else:
            # Use `on` if provided. Otherwise,
            # implicitly use identically named columns as the key columns:
            on_names = (
                _coerce_to_tuple(self.on)
                if self.on is not None
                else set(self.lhs._data.keys()) & set(self.rhs._data.keys())
            )
            left_keys = [
                ColumnView(obj=self.lhs, column=on) for on in on_names
            ]
            right_keys = [
                ColumnView(obj=self.rhs, column=on) for on in on_names
            ]

        if len(left_keys) != len(right_keys):
            raise ValueError(
                "Merge operands must have same number of join key columns"
            )

        self._keys = JoinKeys(left=left_keys, right=right_keys)

    def perform_merge(self):
        self.match_key_dtypes(_input_to_libcudf_castrules_any)

        left_key_indices = [key.get_numeric_index() for key in self._keys.left]
        right_key_indices = [
            key.get_numeric_index() for key in self._keys.right
        ]
        left_rows, right_rows = libcudf.join.join(
            self.lhs,
            self.rhs,
            left_on=left_key_indices,
            right_on=right_key_indices,
            how=self.how,
        )
        return self.construct_result(left_rows, right_rows)

    def construct_result(self, left_rows, right_rows):
        self.match_key_dtypes(_libcudf_to_output_castrules)

        # first construct the index.
        if self.left_index and self.right_index:
            if self.how == "right":
                out_index = self.rhs.index._gather(left_rows, nullify=True)
            else:
                out_index = self.lhs.index._gather(left_rows, nullify=True)
        elif self.left_index:
            # left_index and right_on
            out_index = self.rhs.index._gather(right_rows, nullify=True)
        elif self.right_index:
            # right_index and left_on
            out_index = self.lhs.index._gather(left_rows, nullify=True)
        else:
            out_index = None

        # now construct the data:
        data = cudf.core.column_accessor.ColumnAccessor()
        left_names, right_names = self.output_column_names()

        for lcol in left_names:
            data[left_names[lcol]] = self.lhs._data[lcol].take(
                left_rows, nullify=True
            )
        for rcol in right_names:
            data[right_names[rcol]] = self.rhs._data[rcol].take(
                right_rows, nullify=True
            )

        result = self.out_class._from_data(data, index=out_index)

        # if outer join, key columns are combine:
        if self.how == "outer":
            for lkey, rkey in zip(*self._keys):
                # get the key column as it appears in the result:
                out_key = ColumnView(
                    result, column=lkey.column, index=lkey.index
                )

                # fill nulls in the key column with values from the RHS
                out_key.set_value(
                    out_key.value.fillna(
                        rkey.value.take(right_rows, nullify=True)
                    )
                )

        return self.sort_result(result)

    def sort_result(self, result):
        # If sort=True, Pandas sorts on the key columns in the
        # same order as given in 'on'. If the indices are used as
        # keys, the index will be sorted. If one index is specified,
        # the key columns on the other side will be used to sort.
        if self.sort:
            if self.on:
                return result.sort_values(
                    _coerce_to_list(self.on), ignore_index=True
                )
            elif self.left_index and self.right_index:
                return result.sort_index()
            elif self.left_index:
                return result.sort_values(_coerce_to_list(self.right_on))
            else:
                # self.right_index and self.left_on
                return result.sort_values(_coerce_to_list(self.left_on))
        return result

    def output_column_names(self):
        # Return mappings of input column names to (possibly) suffixed
        # result column names
        left_names = OrderedDict(
            zip(self.lhs._data.keys(), self.lhs._data.keys())
        )
        right_names = OrderedDict(
            zip(self.rhs._data.keys(), self.rhs._data.keys())
        )
        common_names = set(left_names) & set(right_names)

        if self.on:
            key_columns_with_same_name = self.on
        else:
            key_columns_with_same_name = []
            for lkey, rkey in zip(*self._keys):
                if (lkey.is_index_level, rkey.is_index_level) == (
                    False,
                    False,
                ):
                    if lkey.name == rkey.name:
                        key_columns_with_same_name.append(lkey.name)
        for name in common_names:
            if name not in key_columns_with_same_name:
                left_names[name] = f"{name}{self.lsuffix}"
                right_names[name] = f"{name}{self.rsuffix}"
            else:
                del right_names[name]
        return left_names, right_names

    @staticmethod
    def validate_merge_params(
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

    def match_key_dtypes(self, match_func):
        # match the dtypes of the key columns in
        # self.lhs and self.rhs according to the matching
        # function `match_func`
        for left_key, right_key in zip(*self._keys):
            lcol, rcol = left_key.value, right_key.value
            dtype = match_func(lcol, rcol, how=self.how)
            left_key.set_value(lcol.astype(dtype))
            right_key.set_value(rcol.astype(dtype))


class MergeSemi(MergeBase):
    def perform_merge(self):
        self.match_key_dtypes(_input_to_libcudf_castrules_any)

        left_key_indices = [key.get_numeric_index() for key in self._keys.left]
        right_key_indices = [
            key.get_numeric_index() for key in self._keys.right
        ]
        left_rows = libcudf.join.semi_join(
            self.lhs,
            self.rhs,
            left_on=left_key_indices,
            right_on=right_key_indices,
            how=self.how,
        )
        return self.construct_result(left_rows, cudf.core.column.as_column([]))

    def output_column_names(self):
        left_names, _ = super().output_column_names()
        return left_names, {}


def _coerce_to_tuple(obj):
    if hasattr(obj, "__iter__") and not isinstance(obj, str):
        return tuple(obj)
    else:
        return (obj,)


def _coerce_to_list(obj):
    return list(_coerce_to_tuple(obj))
