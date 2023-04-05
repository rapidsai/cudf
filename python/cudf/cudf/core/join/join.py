# Copyright (c) 2020-2022, NVIDIA CORPORATION.
from __future__ import annotations

import warnings
from typing import Any, ClassVar, List, Optional

import cudf
from cudf import _lib as libcudf
from cudf.core.join._join_helpers import (
    _coerce_to_tuple,
    _ColumnIndexer,
    _IndexIndexer,
    _match_join_keys,
)


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
    _joiner: ClassVar[staticmethod] = staticmethod(libcudf.join.join)

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

            left_key.set(self.lhs, lcol_casted, validate=False)
            right_key.set(self.rhs, rcol_casted, validate=False)

        left_rows, right_rows = self._joiner(
            left_join_cols,
            right_join_cols,
            how=self.how,
        )

        gather_kwargs = {
            "nullify": True,
            "check_bounds": False,
            "keep_index": self._using_left_index or self._using_right_index,
        }
        left_result = (
            self.lhs._gather(gather_map=left_rows, **gather_kwargs)
            if left_rows is not None
            else cudf.DataFrame._from_data({})
        )
        right_result = (
            self.rhs._gather(gather_map=right_rows, **gather_kwargs)
            if right_rows is not None
            else cudf.DataFrame._from_data({})
        )

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
                        validate=False,
                    )

        # All columns from the left table make it into the output. Non-key
        # columns that share a name with a column in the right table are
        # suffixed with the provided suffix.
        common_names = set(left_result._data.names) & set(
            right_result._data.names
        )
        cols_to_suffix = common_names - self._key_columns_with_same_name
        data = {
            (f"{name}{self.lsuffix}" if name in cols_to_suffix else name): col
            for name, col in left_result._data.items()
        }

        # The right table follows the same rule as the left table except that
        # key columns from the right table are removed.
        for name, col in right_result._data.items():
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

        index: Optional[cudf.BaseIndex]
        if self._using_right_index:
            # right_index and left_on
            index = left_result._index
        elif self._using_left_index:
            # left_index and right_on
            index = right_result._index
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
        by: List[Any] = []
        if self._using_left_index and self._using_right_index:
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
                check_bounds=False,
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
            warnings.warn(
                "merging between different levels is deprecated and will be "
                f"removed in a future version. ({lhs._data.nlevels} levels on "
                f"the left, {rhs._data.nlevels} on the right)",
                FutureWarning,
            )


class MergeSemi(Merge):
    _joiner: ClassVar[staticmethod] = staticmethod(libcudf.join.semi_join)

    def _merge_results(self, lhs: cudf.DataFrame, rhs: cudf.DataFrame):
        # semi-join result includes only lhs columns
        return lhs._data, lhs._index
