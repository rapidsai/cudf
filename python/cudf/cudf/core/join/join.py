# Copyright (c) 2020-2021, NVIDIA CORPORATION.
from __future__ import annotations

import functools
from collections import namedtuple
from typing import TYPE_CHECKING, Callable, Tuple

import cudf
from cudf import _lib as libcudf
from cudf.core.join._join_helpers import (
    _coerce_to_tuple,
    _frame_select_by_indexers,
    _Indexer,
    _match_join_keys,
)

if TYPE_CHECKING:
    from cudf.core.frame import Frame


def merge(
    lhs,
    rhs,
    *,
    on,
    left_on,
    right_on,
    left_index,
    right_index,
    how,
    sort,
    method,
    indicator,
    suffixes,
):
    if how in {"leftsemi", "leftanti"}:
        merge_cls = MergeSemi
    else:
        merge_cls = Merge
    mergeobj = merge_cls(
        lhs,
        rhs,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        how=how,
        sort=sort,
        method=method,
        indicator=indicator,
        suffixes=suffixes,
    )
    return mergeobj.perform_merge()


_JoinKeys = namedtuple("JoinKeys", ["left", "right"])


class Merge(object):
    # A namedtuple of indexers representing the left and right keys
    _keys: _JoinKeys

    # The joiner function must have the following signature:
    #
    #     def joiner(
    #         lhs: Frame,
    #         rhs: Frame
    #     ) -> Tuple[Optional[Column], Optional[Column]]:
    #          ...
    #
    # where `lhs` and `rhs` are Frames composed of the left and right
    # join key. The `joiner` returns a tuple of two Columns
    # representing the rows to gather from the left- and right- side
    # tables respectively.
    _joiner: Callable

    def __init__(
        self,
        lhs,
        rhs,
        *,
        on,
        left_on,
        right_on,
        left_index,
        right_index,
        how,
        sort,
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
        suffixes : list like
            Left and right suffixes specified together, unpacked into lsuffix
            and rsuffix.
        """
        self._validate_merge_params(
            lhs,
            rhs,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            suffixes=suffixes,
        )
        self._joiner = functools.partial(libcudf.join.join, how=how)

        self.lhs = lhs
        self.rhs = rhs
        self.on = on
        self.left_on = left_on
        self.right_on = right_on
        self.left_index = left_index
        self.right_index = right_index
        self.how = how
        self.sort = sort
        if suffixes:
            self.lsuffix, self.rsuffix = suffixes
        self._compute_join_keys()

    @property
    def _out_class(self):
        # type of the result
        out_class = cudf.DataFrame

        if isinstance(self.lhs, cudf.MultiIndex) or isinstance(
            self.rhs, cudf.MultiIndex
        ):
            out_class = cudf.MultiIndex
        elif isinstance(self.lhs, cudf.Index):
            out_class = self.lhs.__class__
        return out_class

    def perform_merge(self) -> Frame:
        lhs, rhs = self._match_key_dtypes(self.lhs, self.rhs)

        left_table = _frame_select_by_indexers(lhs, self._keys.left)
        right_table = _frame_select_by_indexers(rhs, self._keys.right)

        left_rows, right_rows = self._joiner(
            left_table, right_table, how=self.how,
        )
        lhs, rhs = self._restore_categorical_keys(lhs, rhs)

        left_result = cudf.core.frame.Frame()
        right_result = cudf.core.frame.Frame()

        gather_index = self.left_index or self.right_index
        if left_rows is not None:
            left_result = lhs._gather(
                left_rows, nullify=True, keep_index=gather_index
            )
        if right_rows is not None:
            right_result = rhs._gather(
                right_rows, nullify=True, keep_index=gather_index
            )

        result = self._merge_results(left_result, right_result)

        if self.sort:
            result = self._sort_result(result)
        return result

    def _compute_join_keys(self):
        # Computes self._keys
        left_keys = []
        right_keys = []
        if (
            self.left_index
            or self.right_index
            or self.left_on
            or self.right_on
        ):
            if self.left_index:
                left_keys.extend(
                    [
                        _Indexer(name=on, index=True)
                        for on in self.lhs.index.names
                    ]
                )
            if self.left_on:
                # TODO: require left_on or left_index to be specified
                left_keys.extend(
                    [
                        _Indexer(name=on, column=True)
                        for on in _coerce_to_tuple(self.left_on)
                    ]
                )
            if self.right_index:
                right_keys.extend(
                    [
                        _Indexer(name=on, index=True)
                        for on in self.rhs.index.names
                    ]
                )
            if self.right_on:
                # TODO: require right_on or right_index to be specified
                right_keys.extend(
                    [
                        _Indexer(name=on, column=True)
                        for on in _coerce_to_tuple(self.right_on)
                    ]
                )
        elif self.on:
            on_names = _coerce_to_tuple(self.on)
            for on in on_names:
                # If `on` is provided, Merge on columns if present,
                # otherwise default to indexes.
                if on in self.lhs._data:
                    left_keys.append(_Indexer(name=on, column=True))
                else:
                    left_keys.append(_Indexer(name=on, index=True))
                if on in self.rhs._data:
                    right_keys.append(_Indexer(name=on, column=True))
                else:
                    right_keys.append(_Indexer(name=on, index=True))

        else:
            # if `on` is not provided and we're not merging
            # index with column or on both indexes, then use
            # the intersection  of columns in both frames
            on_names = set(self.lhs._data) & set(self.rhs._data)
            left_keys = [_Indexer(name=on, column=True) for on in on_names]
            right_keys = [_Indexer(name=on, column=True) for on in on_names]

        if len(left_keys) != len(right_keys):
            raise ValueError(
                "Merge operands must have same number of join key columns"
            )

        self._keys = _JoinKeys(left=left_keys, right=right_keys)

    def _merge_results(self, left_result: Frame, right_result: Frame) -> Frame:
        # Merge the Frames `left_result` and `right_result` into a single
        # `Frame`, suffixing column names if necessary.

        # If two key columns have the same name, a single output column appears
        # in the result. For all other join types, the key column from the rhs
        # is simply dropped. For outer joins, the two key columns are combined
        # by filling nulls in the left key column with corresponding values
        # from the right key column:
        if self.how == "outer":
            for lkey, rkey in zip(*self._keys):
                if lkey.name == rkey.name:
                    # fill nulls in lhs from values in the rhs
                    lkey.set(
                        left_result,
                        lkey.get(left_result).fillna(rkey.get(right_result)),
                        validate=False,
                    )

        # Compute the result column names:
        # left_names and right_names will be a mappings of input column names
        # to the corresponding names in the final result.
        left_names = dict(zip(left_result._data, left_result._data))
        right_names = dict(zip(right_result._data, right_result._data))

        # For any columns from left_result and right_result that have the same
        # name:
        # - if they are key columns, keep only the left column
        # - if they are not key columns, use suffixes to differentiate them
        #   in the final result
        common_names = set(left_names) & set(right_names)

        if self.on:
            key_columns_with_same_name = self.on
        else:
            key_columns_with_same_name = [
                lkey.name
                for lkey, rkey in zip(*self._keys)
                if (
                    (lkey.index, rkey.index) == (False, False)
                    and lkey.name == rkey.name
                )
            ]
        for name in common_names:
            if name not in key_columns_with_same_name:
                left_names[name] = f"{name}{self.lsuffix}"
                right_names[name] = f"{name}{self.rsuffix}"
            else:
                del right_names[name]

        # Assemble the data columns of the result:
        data = left_result._data.__class__()

        for lcol in left_names:
            data.set_by_label(
                left_names[lcol], left_result._data[lcol], validate=False
            )
        for rcol in right_names:
            data.set_by_label(
                right_names[rcol], right_result._data[rcol], validate=False
            )

        # Index of the result:
        if self.left_index and self.right_index:
            index = left_result._index
        elif self.left_index:
            # left_index and right_on
            index = right_result._index
        elif self.right_index:
            # right_index and left_on
            index = left_result._index
        else:
            index = None

        # Construct result from data and index:
        result = self._out_class._from_data(data=data, index=index)

        return result

    def _sort_result(self, result: Frame) -> Frame:
        # Pandas sorts on the key columns in the
        # same order as given in 'on'. If the indices are used as
        # keys, the index will be sorted. If one index is specified,
        # the key columns on the other side will be used to sort.
        if self.on:
            if isinstance(result, cudf.Index):
                sort_order = result._get_sorted_inds()
            else:
                # need a list instead of a tuple here because
                # _get_sorted_inds calls down to ColumnAccessor.get_by_label
                # which handles lists and tuples differently
                sort_order = result._get_sorted_inds(
                    list(_coerce_to_tuple(self.on))
                )
            return result._gather(sort_order, keep_index=False)
        by = []
        if self.left_index and self.right_index:
            if result._index is not None:
                by.extend(result._index._data.columns)
        if self.left_on:
            by.extend(
                [result._data[col] for col in _coerce_to_tuple(self.left_on)]
            )
        if self.right_on:
            by.extend(
                [result._data[col] for col in _coerce_to_tuple(self.right_on)]
            )
        if by:
            to_sort = cudf.DataFrame._from_columns(by)
            sort_order = to_sort.argsort()
            result = result._gather(sort_order)
        return result

    @staticmethod
    def _validate_merge_params(
        lhs,
        rhs,
        on,
        left_on,
        right_on,
        left_index,
        right_index,
        how,
        suffixes,
    ):
        """
        Error for various invalid combinations of merge input parameters
        """
        # must actually support the requested merge type
        if how not in {"left", "inner", "outer", "leftanti", "leftsemi"}:
            raise NotImplementedError(f"{how} merge not supported yet")

        if on:
            if left_on or right_on:
                # Passing 'on' with 'left_on' or 'right_on' is ambiguous
                raise ValueError(
                    'Can only pass argument "on" OR "left_on" '
                    'and "right_on", not a combination of both.'
                )
            else:
                # the validity of 'on' being checked by _Indexer
                return

        # Can't merge on unnamed Series
        if (isinstance(lhs, cudf.Series) and not lhs.name) or (
            isinstance(rhs, cudf.Series) and not rhs.name
        ):
            raise ValueError("Can not merge on unnamed Series")

        # If nothing specified, must have common cols to use implicitly
        same_named_columns = set(lhs._data) & set(rhs._data)
        if (
            not (left_index or right_index)
            and not (left_on or right_on)
            and len(same_named_columns) == 0
        ):
            raise ValueError("No common columns to perform merge on")

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

    def _match_key_dtypes(self, lhs: Frame, rhs: Frame) -> Tuple[Frame, Frame]:
        # Match the dtypes of the key columns from lhs and rhs
        out_lhs = lhs.copy(deep=False)
        out_rhs = rhs.copy(deep=False)
        for left_key, right_key in zip(*self._keys):
            lcol, rcol = left_key.get(lhs), right_key.get(rhs)
            lcol_casted, rcol_casted = _match_join_keys(
                lcol, rcol, how=self.how
            )
            if lcol is not lcol_casted:
                left_key.set(out_lhs, lcol_casted, validate=False)
            if rcol is not rcol_casted:
                right_key.set(out_rhs, rcol_casted, validate=False)
        return out_lhs, out_rhs

    def _restore_categorical_keys(
        self, lhs: Frame, rhs: Frame
    ) -> Tuple[Frame, Frame]:
        # For inner joins, any categorical keys in `self.lhs` and `self.rhs`
        # were casted to their category type to produce `lhs` and `rhs`.
        # Here, we cast them back.
        out_lhs = lhs.copy(deep=False)
        out_rhs = rhs.copy(deep=False)
        if self.how == "inner":
            for left_key, right_key in zip(*self._keys):
                if isinstance(
                    left_key.get(self.lhs).dtype, cudf.CategoricalDtype
                ) and isinstance(
                    right_key.get(self.rhs).dtype, cudf.CategoricalDtype
                ):
                    left_key.set(
                        out_lhs,
                        left_key.get(out_lhs).astype("category"),
                        validate=False,
                    )
                    right_key.set(
                        out_rhs,
                        right_key.get(out_rhs).astype("category"),
                        validate=False,
                    )
        return out_lhs, out_rhs


class MergeSemi(Merge):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._joiner = functools.partial(
            libcudf.join.semi_join, how=kwargs["how"]
        )

    def _merge_results(self, lhs: Frame, rhs: Frame) -> Frame:
        # semi-join result includes only lhs columns
        if issubclass(self._out_class, cudf.Index):
            return self._out_class._from_data(lhs._data)
        else:
            return self._out_class._from_data(lhs._data, index=lhs._index)
