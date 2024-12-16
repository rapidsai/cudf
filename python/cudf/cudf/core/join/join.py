# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from __future__ import annotations

from typing import Any

import pylibcudf as plc

import cudf
from cudf import _lib as libcudf
from cudf._lib.types import size_type_dtype
from cudf.core._internals import sorting
from cudf.core.buffer import acquire_spill_lock
from cudf.core.copy_types import GatherMap
from cudf.core.join._join_helpers import (
    _coerce_to_tuple,
    _ColumnIndexer,
    _IndexIndexer,
    _match_join_keys,
)


class Merge:
    @staticmethod
    @acquire_spill_lock()
    def _joiner(
        lhs: list[libcudf.column.Column],
        rhs: list[libcudf.column.Column],
        how: str,
    ) -> tuple[libcudf.column.Column, libcudf.column.Column]:
        if how == "outer":
            how = "full"
        if (join_func := getattr(plc.join, f"{how}_join", None)) is None:
            raise ValueError(f"Invalid join type {how}")

        left_rows, right_rows = join_func(
            plc.Table([col.to_pylibcudf(mode="read") for col in lhs]),
            plc.Table([col.to_pylibcudf(mode="read") for col in rhs]),
            plc.types.NullEquality.EQUAL,
        )
        return libcudf.column.Column.from_pylibcudf(
            left_rows
        ), libcudf.column.Column.from_pylibcudf(right_rows)

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
        lhs : DataFrame
            The left operand of the merge
        rhs : DataFrame
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
            Boolean flag indicating the right index column or columns
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

        self.lhs = lhs.copy(deep=False)
        self.rhs = rhs.copy(deep=False)
        self.how = how
        # If the user requests that the result is sorted or we're in
        # pandas-compatible mode we have various obligations on the
        # output order:
        #
        # compat-> | False                    | True
        # sort     |                          |
        # ---------+--------------------------+-------------------------------
        #     False| no obligation            | ordering as per pandas docs(*)
        #     True | sorted lexicographically | sorted lexicographically(*)
        #
        # (*) If two keys are equal, tiebreak is to use input table order.
        #
        # In pandas-compat mode, we have obligations on the order to
        # match pandas (even if sort=False), see
        # pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html.
        # The ordering requirements differ depending on which join
        # type is specified:
        #
        # - left: preserve key order (only keeping left keys)
        # - right: preserve key order (only keeping right keys)
        # - inner: preserve key order (of left keys)
        # - outer: sort keys lexicographically
        # - cross (not supported): preserve key order (of left keys)
        #
        # Moreover, in all cases, whenever there is a tiebreak
        # situation (for sorting or otherwise), the deciding order is
        # "input table order"
        self.sort = sort or (
            cudf.get_option("mode.pandas_compatible") and how == "outer"
        )
        self.preserve_key_order = cudf.get_option(
            "mode.pandas_compatible"
        ) and how in {
            "inner",
            "outer",
            "left",
            "right",
        }
        self.lsuffix, self.rsuffix = suffixes

        # At this point validation guarantees that if on is not None we
        # don't have any other args, so we can apply it directly to left_on and
        # right_on.
        self._using_left_index = bool(left_index)
        left_on = (
            lhs.index._column_names
            if left_index
            else left_on
            if left_on
            else on
        )
        self._using_right_index = bool(right_index)
        right_on = (
            rhs.index._column_names
            if right_index
            else right_on
            if right_on
            else on
        )

        if left_on or right_on:
            self._left_keys = [
                _ColumnIndexer(name=on)
                if not self._using_left_index and on in lhs._data
                else _IndexIndexer(name=on)
                for on in (_coerce_to_tuple(left_on) if left_on else [])
            ]
            self._right_keys = [
                _ColumnIndexer(name=on)
                if not self._using_right_index and on in rhs._data
                else _IndexIndexer(name=on)
                for on in (_coerce_to_tuple(right_on) if right_on else [])
            ]
            if len(self._left_keys) != len(self._right_keys):
                raise ValueError(
                    "Merge operands must have same number of join key columns"
                )
            self._using_left_index = any(
                isinstance(idx, _IndexIndexer) for idx in self._left_keys
            )
            self._using_right_index = any(
                isinstance(idx, _IndexIndexer) for idx in self._right_keys
            )
        else:
            # if `on` is not provided and we're not merging
            # index with column or on both indexes, then use
            # the intersection  of columns in both frames
            on_names = set(lhs._data) & set(rhs._data)
            self._left_keys = [_ColumnIndexer(name=on) for on in on_names]
            self._right_keys = [_ColumnIndexer(name=on) for on in on_names]
            self._using_left_index = False
            self._using_right_index = False

        self._key_columns_with_same_name = (
            set(_coerce_to_tuple(on))
            if on
            else {
                lkey.name
                for lkey, rkey in zip(self._left_keys, self._right_keys)
                if lkey.name == rkey.name
                and not (
                    isinstance(lkey, _IndexIndexer)
                    or isinstance(rkey, _IndexIndexer)
                )
            }
        )

    def _gather_maps(self, left_cols, right_cols):
        # Produce gather maps for the join, optionally reordering to
        # match pandas-order in compat mode.
        maps = self._joiner(
            left_cols,
            right_cols,
            how=self.how,
        )
        if not self.preserve_key_order:
            return maps
        # We should only get here if we're in a join on which
        # pandas-compat places some ordering obligation (which
        # precludes a semi-join)
        # We must perform this reordering even if sort=True since the
        # obligation to ensure tiebreaks appear in input table order
        # means that the gather maps must be permuted into an original
        # order.
        assert self.how in {"inner", "outer", "left", "right"}
        # And hence both maps returned from the libcudf join should be
        # non-None.
        assert all(m is not None for m in maps)
        lengths = [len(left_cols[0]), len(right_cols[0])]
        # Only nullify those maps that need it.
        nullify = [
            self.how not in {"inner", "left"},
            self.how not in {"inner", "right"},
        ]
        # To reorder maps so that they are in order of the input
        # tables, we gather from iota on both right and left, and then
        # sort the gather maps with those two columns as key.
        key_order = [
            cudf.core.column.as_column(range(n), dtype=size_type_dtype).take(
                map_, nullify=null, check_bounds=False
            )
            for map_, n, null in zip(maps, lengths, nullify)
        ]
        return sorting.sort_by_key(
            list(maps),
            # If how is right, right map is primary sort key.
            key_order[:: -1 if self.how == "right" else 1],
            [True] * len(key_order),
            ["last"] * len(key_order),
            stable=True,
        )

    def perform_merge(self) -> cudf.DataFrame:
        left_join_cols = []
        right_join_cols = []

        for left_key, right_key in zip(self._left_keys, self._right_keys):
            lcol = left_key.get(self.lhs)
            rcol = right_key.get(self.rhs)
            lcol_casted, rcol_casted = _match_join_keys(lcol, rcol, self.how)
            left_join_cols.append(lcol_casted)
            right_join_cols.append(rcol_casted)

            # Categorical dtypes must be cast back from the underlying codes
            # type that was returned by _match_join_keys.
            if (
                self.how == "inner"
                and isinstance(lcol.dtype, cudf.CategoricalDtype)
                and isinstance(rcol.dtype, cudf.CategoricalDtype)
            ):
                lcol_casted = lcol_casted.astype("category")
                rcol_casted = rcol_casted.astype("category")

            left_key.set(self.lhs, lcol_casted)
            right_key.set(self.rhs, rcol_casted)

        left_rows, right_rows = self._gather_maps(
            left_join_cols, right_join_cols
        )
        gather_kwargs = {
            "keep_index": self._using_left_index or self._using_right_index,
        }
        left_result = (
            self.lhs._gather(
                GatherMap.from_column_unchecked(
                    left_rows, len(self.lhs), nullify=True
                ),
                **gather_kwargs,
            )
            if left_rows is not None
            else cudf.DataFrame._from_data({})
        )
        del left_rows
        right_result = (
            self.rhs._gather(
                GatherMap.from_column_unchecked(
                    right_rows, len(self.rhs), nullify=True
                ),
                **gather_kwargs,
            )
            if right_rows is not None
            else cudf.DataFrame._from_data({})
        )
        del right_rows
        result = cudf.DataFrame._from_data(
            *self._merge_results(left_result, right_result)
        )

        if self.sort:
            result = self._sort_result(result)
        return result

    def _merge_results(
        self, left_result: cudf.DataFrame, right_result: cudf.DataFrame
    ):
        # Merge the DataFrames `left_result` and `right_result` into a single
        # `DataFrame`, suffixing column names if necessary.

        # If two key columns have the same name, a single output column appears
        # in the result. For all non-outer join types, the key column from the
        # rhs is simply dropped. For outer joins, the two key columns are
        # combined by filling nulls in the left key column with corresponding
        # values from the right key column:
        if self.how == "outer":
            for lkey, rkey in zip(self._left_keys, self._right_keys):
                if lkey.name == rkey.name:
                    # fill nulls in lhs from values in the rhs
                    lkey.set(
                        left_result,
                        lkey.get(left_result).fillna(rkey.get(right_result)),
                    )

        # All columns from the left table make it into the output. Non-key
        # columns that share a name with a column in the right table are
        # suffixed with the provided suffix.
        common_names = set(left_result._column_names) & set(
            right_result._column_names
        )
        cols_to_suffix = common_names - self._key_columns_with_same_name
        data = {
            (f"{name}{self.lsuffix}" if name in cols_to_suffix else name): col
            for name, col in left_result._column_labels_and_values
        }

        # The right table follows the same rule as the left table except that
        # key columns from the right table are removed.
        for name, col in right_result._column_labels_and_values:
            if name in common_names:
                if name not in self._key_columns_with_same_name:
                    data[f"{name}{self.rsuffix}"] = col
            else:
                data[name] = col

        # determine if the result has multiindex columns.  The result
        # of a join has a MultiIndex as its columns if:
        # - both the `lhs` and `rhs` have a MultiIndex columns
        # OR
        # - either one of `lhs` or `rhs` have a MultiIndex columns,
        #   and the other is empty (i.e., no columns)
        if self.lhs._data and self.rhs._data:
            multiindex_columns = (
                self.lhs._data.multiindex and self.rhs._data.multiindex
            )
        elif self.lhs._data:
            multiindex_columns = self.lhs._data.multiindex
        elif self.rhs._data:
            multiindex_columns = self.rhs._data.multiindex
        else:
            multiindex_columns = False

        index: cudf.BaseIndex | None
        if self._using_right_index:
            # right_index and left_on
            index = left_result.index
        elif self._using_left_index:
            # left_index and right_on
            index = right_result.index
        else:
            index = None

        # Construct result from data and index:
        return (
            left_result._data.__class__(
                data=data, multiindex=multiindex_columns
            ),
            index,
        )

    def _sort_result(self, result: cudf.DataFrame) -> cudf.DataFrame:
        # Pandas sorts on the key columns in the
        # same order as given in 'on'. If the indices are used as
        # keys, the index will be sorted. If one index is specified,
        # the key columns on the other side will be used to sort.
        # In pandas-compatible mode, tie-breaking for multiple equal
        # sort keys is to produce output in input dataframe order.
        # This is taken care of by using a stable sort here, and (in
        # pandas-compat mode) reordering the gather maps before
        # producing the input result.
        by: list[Any] = []
        if self._using_left_index and self._using_right_index:
            by.extend(result.index._columns)
        if not self._using_left_index:
            by.extend([result._data[col.name] for col in self._left_keys])
        if not self._using_right_index:
            by.extend([result._data[col.name] for col in self._right_keys])
        if by:
            keep_index = self._using_left_index or self._using_right_index
            if keep_index:
                to_sort = [*result.index._columns, *result._columns]
                index_names = result.index.names
            else:
                to_sort = [*result._columns]
                index_names = None
            result_columns = sorting.sort_by_key(
                to_sort,
                by,
                [True] * len(by),
                ["last"] * len(by),
                stable=True,
            )
            result = result._from_columns_like_self(
                result_columns, result._column_names, index_names
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
        # Error for various invalid combinations of merge input parameters

        # We must actually support the requested merge type
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

        if (
            isinstance(lhs, cudf.DataFrame)
            and isinstance(rhs, cudf.DataFrame)
            # An empty column is considered to have 1 level by pandas (can be
            # seen by using lhs.columns.nlevels, but we don't want to use
            # columns internally because it's expensive).
            # TODO: Investigate whether ColumnAccessor.nlevels should be
            # modified in the size 0 case.
            and max(lhs._data.nlevels, 1) != max(rhs._data.nlevels, 1)
        ):
            raise ValueError(
                "Not allowed to merge between different levels. "
                f"({lhs._data.nlevels} levels on "
                f"the left, {rhs._data.nlevels} on the right)"
            )


class MergeSemi(Merge):
    @staticmethod
    @acquire_spill_lock()
    def _joiner(
        lhs: list[libcudf.column.Column],
        rhs: list[libcudf.column.Column],
        how: str,
    ) -> tuple[libcudf.column.Column, None]:
        if (
            join_func := getattr(
                plc.join, f"{how.replace('left', 'left_')}_join", None
            )
        ) is None:
            raise ValueError(f"Invalid join type {how}")

        return libcudf.column.Column.from_pylibcudf(
            join_func(
                plc.Table([col.to_pylibcudf(mode="read") for col in lhs]),
                plc.Table([col.to_pylibcudf(mode="read") for col in rhs]),
                plc.types.NullEquality.EQUAL,
            )
        ), None

    def _merge_results(self, lhs: cudf.DataFrame, rhs: cudf.DataFrame):
        # semi-join result includes only lhs columns
        return lhs._data, lhs.index
