# Copyright (c) 2020-2021, NVIDIA CORPORATION.
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import cudf
from cudf import _lib as libcudf
from cudf.core.join._join_helpers import (
    _coerce_to_tuple,
    _Indexer,
    _match_join_keys,
)

if TYPE_CHECKING:
    from cudf.core.frame import Frame


class Merge:
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
    _joiner: Callable = libcudf.join.join

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

        self.how = how
        self.sort = sort
        self.lsuffix, self.rsuffix = suffixes

        # At this point validation guarantees that if on is not None we
        # don't have any other args, so we can apply it directly to left_on and
        # right_on.
        self._using_left_index = bool(left_index)
        left_on = (
            lhs.index._data.names if left_index else left_on if left_on else on
        )
        self._using_right_index = bool(right_index)
        right_on = (
            rhs.index._data.names
            if right_index
            else right_on
            if right_on
            else on
        )

        if left_on or right_on:
            self._left_keys = [
                _Indexer(name=on, column=True)
                if not self._using_left_index and on in lhs._data
                else _Indexer(name=on, index=True)
                for on in (_coerce_to_tuple(left_on) if left_on else [])
            ]
            self._right_keys = [
                _Indexer(name=on, column=True)
                if not self._using_right_index and on in rhs._data
                else _Indexer(name=on, index=True)
                for on in (_coerce_to_tuple(right_on) if right_on else [])
            ]
            if len(self._left_keys) != len(self._right_keys):
                raise ValueError(
                    "Merge operands must have same number of join key columns"
                )
        else:
            # if `on` is not provided and we're not merging
            # index with column or on both indexes, then use
            # the intersection  of columns in both frames
            on_names = set(lhs._data) & set(rhs._data)
            self._left_keys = [
                _Indexer(name=on, column=True) for on in on_names
            ]
            self._right_keys = [
                _Indexer(name=on, column=True) for on in on_names
            ]

        self.output_lhs = lhs.copy(deep=False)
        self.output_rhs = rhs.copy(deep=False)

        left_join_cols = {}
        right_join_cols = {}

        for left_key, right_key in zip(self._left_keys, self._right_keys):
            lcol = left_key.get(self.output_lhs)
            rcol = right_key.get(self.output_rhs)
            lcol_casted, rcol_casted = _match_join_keys(lcol, rcol, self.how)
            left_join_cols[left_key.name] = lcol_casted
            right_join_cols[left_key.name] = rcol_casted

            # Categorical dtypes must be cast back from the underlying codes
            # type that was returned by _match_join_keys.
            if (
                self.how == "inner"
                and isinstance(lcol.dtype, cudf.CategoricalDtype)
                and isinstance(rcol.dtype, cudf.CategoricalDtype)
            ):
                lcol_casted = lcol_casted.astype("category")
                rcol_casted = rcol_casted.astype("category")

            left_key.set(self.output_lhs, lcol_casted, validate=False)
            right_key.set(self.output_rhs, rcol_casted, validate=False)

        self._left_join_table = cudf.core.frame.Frame(left_join_cols)
        self._right_join_table = cudf.core.frame.Frame(right_join_cols)

        if isinstance(lhs, cudf.MultiIndex) or isinstance(
            rhs, cudf.MultiIndex
        ):
            self._out_class = cudf.MultiIndex
        elif isinstance(lhs, cudf.BaseIndex):
            self._out_class = lhs.__class__
        else:
            self._out_class = cudf.DataFrame

        self._key_columns_with_same_name = (
            on
            if on
            else []
            if (self._using_left_index or self._using_right_index)
            else [
                lkey.name
                for lkey, rkey in zip(self._left_keys, self._right_keys)
                if lkey.name == rkey.name
            ]
        )

    def perform_merge(self) -> Frame:
        left_rows, right_rows = self._joiner(
            self._left_join_table, self._right_join_table, how=self.how,
        )

        left_result = cudf.core.frame.Frame()
        right_result = cudf.core.frame.Frame()

        gather_index = self._using_left_index or self._using_right_index
        if left_rows is not None:
            left_result = self.output_lhs._gather(
                left_rows, nullify=True, keep_index=gather_index
            )
        if right_rows is not None:
            right_result = self.output_rhs._gather(
                right_rows, nullify=True, keep_index=gather_index
            )

        result = self._out_class._from_data(
            *self._merge_results(left_result, right_result)
        )

        if self.sort:
            result = self._sort_result(result)
        return result

    def _merge_results(self, left_result: Frame, right_result: Frame):
        # Merge the Frames `left_result` and `right_result` into a single
        # `Frame`, suffixing column names if necessary.

        # If two key columns have the same name, a single output column appears
        # in the result. For all other join types, the key column from the rhs
        # is simply dropped. For outer joins, the two key columns are combined
        # by filling nulls in the left key column with corresponding values
        # from the right key column:
        # TODO: Move this to the creation of the output_lhs in the constructor
        # as well.
        if self.how == "outer":
            for lkey, rkey in zip(self._left_keys, self._right_keys):
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
        common_names = set(left_result._data.names) & set(
            right_result._data.names
        )

        for name in common_names:
            if name not in self._key_columns_with_same_name:
                left_names[name] = f"{name}{self.lsuffix}"
                right_names[name] = f"{name}{self.rsuffix}"
            else:
                del right_names[name]

        # Assemble the data columns of the result:
        data = {
            **{
                new_name: left_result._data[orig_name]
                for orig_name, new_name in left_names.items()
            },
            **{
                new_name: right_result._data[orig_name]
                for orig_name, new_name in right_names.items()
            },
        }

        # TODO: There is a bug here, we actually need to pull the index columns
        # from both if both left_index and right_index were True.
        if self._using_right_index:
            # right_index and left_on
            index = left_result._index
        elif self._using_left_index:
            # left_index and right_on
            index = right_result._index
        else:
            index = None

        # Construct result from data and index:
        return data, index

    def _sort_result(self, result: Frame) -> Frame:
        # Pandas sorts on the key columns in the
        # same order as given in 'on'. If the indices are used as
        # keys, the index will be sorted. If one index is specified,
        # the key columns on the other side will be used to sort.
        by = []
        if self._using_left_index and self._using_right_index:
            if result._index is not None:
                by.extend(result._index._data.columns)
        if not self._using_left_index:
            by.extend([result._data[col.name] for col in self._left_keys])
        if not self._using_right_index:
            by.extend([result._data[col.name] for col in self._right_keys])
        if by:
            to_sort = cudf.DataFrame._from_data(dict(enumerate(by)))
            sort_order = to_sort.argsort()
            result = result._gather(
                sort_order,
                keep_index=self._using_left_index or self._using_right_index,
            )
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
            elif left_index or right_index:
                # Passing 'on' with 'left_index' or 'right_index' is ambiguous
                raise ValueError(
                    'Can only pass argument "on" OR "left_index" '
                    'and "right_index", not a combination of both.'
                )
            else:
                # the validity of 'on' being checked by _Indexer
                return
        elif left_on and left_index:
            raise ValueError(
                'Can only pass argument "left_on" OR "left_index" not both.'
            )
        elif right_on and right_index:
            raise ValueError(
                'Can only pass argument "right_on" OR "right_index" not both.'
            )

        # Can't merge on a column name that is present in both a frame and its
        # indexes.
        if on:
            for key in on:
                if (key in lhs._data and key in lhs.index._data) or (
                    key in rhs._data and key in rhs.index._data
                ):
                    raise ValueError(
                        f"{key} is both an index level and a "
                        "column label, which is ambiguous."
                    )
        if left_on:
            for key in left_on:
                if key in lhs._data and key in lhs.index._data:
                    raise ValueError(
                        f"{key} is both an index level and a "
                        "column label, which is ambiguous."
                    )
        if right_on:
            for key in right_on:
                if key in rhs._data and key in rhs.index._data:
                    raise ValueError(
                        f"{key} is both an index level and a "
                        "column label, which is ambiguous."
                    )

        # Can't merge on unnamed Series
        if (isinstance(lhs, cudf.Series) and not lhs.name) or (
            isinstance(rhs, cudf.Series) and not rhs.name
        ):
            raise ValueError("Cannot merge on unnamed Series")

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


class MergeSemi(Merge):
    _joiner: Callable = libcudf.join.semi_join

    def _merge_results(self, lhs: Frame, rhs: Frame):
        # semi-join result includes only lhs columns
        return (
            lhs._data,
            lhs._index
            if not issubclass(self._out_class, cudf.Index)
            else None,
        )
