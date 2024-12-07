# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from __future__ import annotations

import copy
import itertools
import pickle
import textwrap
import warnings
from collections import abc
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import cupy as cp
import numpy as np
import pandas as pd

import pylibcudf as plc

import cudf
from cudf import _lib as libcudf
from cudf._lib import groupby as libgroupby
from cudf._lib.sort import segmented_sort_by_key
from cudf._lib.types import size_type_dtype
from cudf.api.extensions import no_default
from cudf.api.types import is_list_like, is_numeric_dtype
from cudf.core._compat import PANDAS_LT_300
from cudf.core.abc import Serializable
from cudf.core.buffer import acquire_spill_lock
from cudf.core.column.column import ColumnBase, StructDtype, as_column
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.copy_types import GatherMap
from cudf.core.join._join_helpers import _match_join_keys
from cudf.core.mixins import Reducible, Scannable
from cudf.core.multiindex import MultiIndex
from cudf.core.udf.groupby_utils import _can_be_jitted, jit_groupby_apply
from cudf.utils.performance_tracking import _performance_tracking
from cudf.utils.utils import GetAttrGetItemMixin

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cudf._typing import (
        AggType,
        DataFrameOrSeries,
        MultiColumnAggType,
        ScalarLike,
    )


def _deprecate_collect():
    warnings.warn(
        "Groupby.collect is deprecated and "
        "will be removed in a future version. "
        "Use `.agg(list)` instead.",
        FutureWarning,
    )


# The three functions below return the quantiles [25%, 50%, 75%]
# respectively, which are called in the describe() method to output
# the summary stats of a GroupBy object
def _quantile_25(x):
    return x.quantile(0.25)


def _quantile_50(x):
    return x.quantile(0.50)


def _quantile_75(x):
    return x.quantile(0.75)


def _is_row_of(chunk, obj):
    return (
        isinstance(chunk, cudf.Series)
        and isinstance(obj, cudf.DataFrame)
        and len(chunk.index) == len(obj._column_names)
        and (chunk.index.to_pandas() == pd.Index(obj._column_names)).all()
    )


NamedAgg = pd.NamedAgg


NamedAgg.__doc__ = """
Helper for column specific aggregation with control over output column names.

Subclass of typing.NamedTuple.

Parameters
----------
column : Hashable
    Column label in the DataFrame to apply aggfunc.
aggfunc : function or str
    Function to apply to the provided column.

Examples
--------
>>> df = cudf.DataFrame({"key": [1, 1, 2], "a": [-1, 0, 1], 1: [10, 11, 12]})
>>> agg_a = cudf.NamedAgg(column="a", aggfunc="min")
>>> agg_1 = cudf.NamedAgg(column=1, aggfunc=lambda x: x.mean())
>>> df.groupby("key").agg(result_a=agg_a, result_1=agg_1)
        result_a  result_1
key
1          -1      10.5
2           1      12.0
"""


groupby_doc_template = textwrap.dedent(
    """Group using a mapper or by a Series of columns.

A groupby operation involves some combination of splitting the object,
applying a function, and combining the results. This can be used to
group large amounts of data and compute operations on these groups.

Parameters
----------
by : mapping, function, label, or list of labels
    Used to determine the groups for the groupby. If by is a
    function, it's called on each value of the object's index.
    If a dict or Series is passed, the Series or dict VALUES will
    be used to determine the groups (the Series' values are first
    aligned; see .align() method). If an cupy array is passed, the
    values are used as-is determine the groups. A label or list
    of labels may be passed to group by the columns in self.
    Notice that a tuple is interpreted as a (single) key.
level : int, level name, or sequence of such, default None
    If the axis is a MultiIndex (hierarchical), group by a particular
    level or levels.
as_index : bool, default True
    For aggregated output, return object with group labels as
    the index. Only relevant for DataFrame input.
    as_index=False is effectively "SQL-style" grouped output.
sort : bool, default False
    Sort result by group key. Differ from Pandas, cudf defaults to
    ``False`` for better performance. Note this does not influence
    the order of observations within each group. Groupby preserves
    the order of rows within each group.
group_keys : bool, optional
    When calling apply and the ``by`` argument produces a like-indexed
    result, add group keys to index to identify pieces. By default group
    keys are not included when the result's index (and column) labels match
    the inputs, and are included otherwise. This argument has no effect if
    the result produced is not like-indexed with respect to the input.
{ret}
Examples
--------
**Series**

>>> ser = cudf.Series([390., 350., 30., 20.],
...                 index=['Falcon', 'Falcon', 'Parrot', 'Parrot'],
...                 name="Max Speed")
>>> ser
Falcon    390.0
Falcon    350.0
Parrot     30.0
Parrot     20.0
Name: Max Speed, dtype: float64
>>> ser.groupby(level=0, sort=True).mean()
Falcon    370.0
Parrot     25.0
Name: Max Speed, dtype: float64
>>> ser.groupby(ser > 100, sort=True).mean()
Max Speed
False     25.0
True     370.0
Name: Max Speed, dtype: float64

**DataFrame**

>>> import cudf
>>> import pandas as pd
>>> df = cudf.DataFrame({{
...     'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot'],
...     'Max Speed': [380., 370., 24., 26.],
... }})
>>> df
   Animal  Max Speed
0  Falcon      380.0
1  Falcon      370.0
2  Parrot       24.0
3  Parrot       26.0
>>> df.groupby(['Animal'], sort=True).mean()
        Max Speed
Animal
Falcon      375.0
Parrot       25.0

>>> arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],
...           ['Captive', 'Wild', 'Captive', 'Wild']]
>>> index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))
>>> df = cudf.DataFrame({{'Max Speed': [390., 350., 30., 20.]}},
...     index=index)
>>> df
                Max Speed
Animal Type
Falcon Captive      390.0
        Wild         350.0
Parrot Captive       30.0
        Wild          20.0
>>> df.groupby(level=0, sort=True).mean()
        Max Speed
Animal
Falcon      370.0
Parrot       25.0
>>> df.groupby(level="Type", sort=True).mean()
        Max Speed
Type
Captive      210.0
Wild         185.0

>>> df = cudf.DataFrame({{'A': 'a a b'.split(),
...                      'B': [1,2,3],
...                      'C': [4,6,5]}})
>>> g1 = df.groupby('A', group_keys=False, sort=True)
>>> g2 = df.groupby('A', group_keys=True, sort=True)

Notice that ``g1`` have ``g2`` have two groups, ``a`` and ``b``, and only
differ in their ``group_keys`` argument. Calling `apply` in various ways,
we can get different grouping results:

>>> g1[['B', 'C']].apply(lambda x: x / x.sum())
          B    C
0  0.333333  0.4
1  0.666667  0.6
2  1.000000  1.0

In the above, the groups are not part of the index. We can have them included
by using ``g2`` where ``group_keys=True``:

>>> g2[['B', 'C']].apply(lambda x: x / x.sum())
            B    C
A
a 0  0.333333  0.4
  1  0.666667  0.6
b 2  1.000000  1.0
"""
)


class GroupBy(Serializable, Reducible, Scannable):
    obj: "cudf.core.indexed_frame.IndexedFrame"

    _VALID_REDUCTIONS = {
        "sum",
        "prod",
        "idxmin",
        "idxmax",
        "min",
        "max",
        "mean",
        "median",
        "nunique",
        "first",
        "last",
        "var",
        "std",
    }

    _VALID_SCANS = {
        "cumsum",
        "cummin",
        "cummax",
    }

    # Necessary because the function names don't directly map to the docs.
    _SCAN_DOCSTRINGS = {
        "cumsum": {"op_name": "Cumulative sum"},
        "cummin": {"op_name": "Cumulative min"},
        "cummax": {"op_name": "Cumulative max"},
    }

    _MAX_GROUPS_BEFORE_WARN = 100

    def __init__(
        self,
        obj,
        by=None,
        level=None,
        sort=False,
        as_index=True,
        dropna=True,
        group_keys=True,
    ):
        """
        Group a DataFrame or Series by a set of columns.

        Parameters
        ----------
        by : optional
            Specifies the grouping columns. Can be any of the following:
            - A Python function called on each value of the object's index
            - A dict or Series that maps index labels to group names
            - A cudf.Index object
            - A str indicating a column name
            - An array of the same length as the object
            - A Grouper object
            - A list of the above
        level : int, level_name or list, optional
            For objects with a MultiIndex, `level` can be used to specify
            grouping by one or more levels of the MultiIndex.
        sort : bool, default False
            Sort the result by group keys. Differ from Pandas, cudf defaults
            to False for better performance.
        as_index : bool, optional
            If as_index=True (default), the group names appear
            as the keys of the resulting DataFrame.
            If as_index=False, the groups are returned as ordinary
            columns of the resulting DataFrame, *if they are named columns*.
        dropna : bool, optional
            If True (default), do not include the "null" group.
        """
        self.obj = obj
        self._as_index = as_index
        self._by = by.copy(deep=True) if isinstance(by, _Grouping) else by
        self._level = level
        self._sort = sort
        self._dropna = dropna
        self._group_keys = group_keys

        if isinstance(self._by, _Grouping):
            self._by._obj = self.obj
            self.grouping = self._by
        else:
            self.grouping = _Grouping(obj, self._by, level)

    def __iter__(self):
        group_names, offsets, _, grouped_values = self._grouped()
        if isinstance(group_names, cudf.BaseIndex):
            group_names = group_names.to_pandas()
        for i, name in enumerate(group_names):
            yield (
                (name,)
                if isinstance(self._by, list) and len(self._by) == 1
                else name,
                grouped_values[offsets[i] : offsets[i + 1]],
            )

    def __len__(self) -> int:
        return self.ngroups

    @property
    def ngroups(self) -> int:
        _, offsets, _, _ = self._grouped()
        return len(offsets) - 1

    @property
    def ndim(self) -> int:
        return self.obj.ndim

    @property
    def dtypes(self):
        """
        Return the dtypes in this group.

        .. deprecated:: 24.04
           Use `.dtypes` on base object instead.

        Returns
        -------
        pandas.DataFrame
            The data type of each column of the group.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 3], 'b': ['x', 'y', 'z', 'a'],
        ...                      'c':[10, 11, 12, 12]})
        >>> df.groupby("a").dtypes
               a       b      c
        a
        1  int64  object  int64
        2  int64  object  int64
        3  int64  object  int64
        """
        warnings.warn(
            f"{type(self).__name__}.dtypes is deprecated and will be "
            "removed in a future version. Check the dtypes on the "
            "base object instead",
            FutureWarning,
        )
        index = self.grouping.keys.unique().sort_values().to_pandas()
        return pd.DataFrame(
            {name: [dtype] * len(index) for name, dtype in self.obj._dtypes},
            index=index,
        )

    @cached_property
    def groups(self):
        """
        Returns a dictionary mapping group keys to row labels.
        """
        group_names, offsets, _, grouped_values = self._grouped()
        grouped_index = grouped_values.index

        if len(group_names) > self._MAX_GROUPS_BEFORE_WARN:
            warnings.warn(
                f"GroupBy.groups() performance scales poorly with "
                f"number of groups. Got {len(group_names)} groups."
            )

        return dict(
            zip(group_names.to_pandas(), grouped_index._split(offsets[1:-1]))
        )

    @cached_property
    def indices(self) -> dict[ScalarLike, cp.ndarray]:
        """
        Dict {group name -> group indices}.

        Examples
        --------
        >>> import cudf
        >>> data = [[10, 20, 30], [10, 30, 40], [40, 50, 30]]
        >>> df = cudf.DataFrame(data, columns=["a", "b", "c"])
        >>> df
            a   b   c
        0  10  20  30
        1  10  30  40
        2  40  50  30
        >>> df.groupby(by=["a"]).indices
        {10: array([0, 1]), 40: array([2])}
        """
        offsets, group_keys, (indices,) = self._groupby.groups(
            [
                cudf.core.column.as_column(
                    range(len(self.obj)), dtype=size_type_dtype
                )
            ]
        )

        group_keys = libcudf.stream_compaction.drop_duplicates(group_keys)
        if len(group_keys) > 1:
            index = cudf.MultiIndex.from_arrays(group_keys)
        else:
            index = cudf.Index._from_column(group_keys[0])
        return dict(
            zip(index.to_pandas(), cp.split(indices.values, offsets[1:-1]))
        )

    @_performance_tracking
    def get_group(self, name, obj=None):
        """
        Construct DataFrame from group with provided name.

        Parameters
        ----------
        name : object
            The name of the group to get as a DataFrame.
        obj : DataFrame, default None
            The DataFrame to take the DataFrame out of.  If
            it is None, the object groupby was called on will
            be used.

        Returns
        -------
        group : same type as obj

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]})
        >>> df
           X  Y
        0  A  1
        1  B  4
        2  A  3
        3  B  2
        >>> df.groupby("X").get_group("A")
           X  Y
        0  A  1
        2  A  3
        """
        if obj is None:
            obj = self.obj
        else:
            warnings.warn(
                "obj is deprecated and will be removed in a future version. "
                "Use ``df.iloc[gb.indices.get(name)]`` "
                "instead of ``gb.get_group(name, obj=df)``.",
                FutureWarning,
            )
        if is_list_like(self._by):
            if isinstance(name, tuple) and len(name) == 1:
                name = name[0]
            else:
                raise KeyError(name)
        return obj.iloc[self.indices[name]]

    @_performance_tracking
    def size(self):
        """
        Return the size of each group.
        """
        col = cudf.core.column.column_empty(
            len(self.obj), "int8", masked=False
        )
        result = (
            cudf.Series._from_column(col, name=getattr(self.obj, "name", None))
            .groupby(self.grouping, sort=self._sort, dropna=self._dropna)
            .agg("size")
        )
        if not self._as_index:
            result = result.rename("size").reset_index()
        return result

    @_performance_tracking
    def cumcount(self, ascending: bool = True):
        """
        Return the cumulative count of keys in each group.

        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from length of group - 1 to 0.
            Currently not supported
        """
        if ascending is not True:
            raise NotImplementedError(
                "ascending is currently not implemented."
            )
        return (
            cudf.Series._from_column(
                cudf.core.column.column_empty(
                    len(self.obj), "int8", masked=False
                ),
                index=self.obj.index,
            )
            .groupby(self.grouping, sort=self._sort)
            .agg("cumcount")
        )

    @_performance_tracking
    def rank(
        self,
        method="average",
        ascending=True,
        na_option="keep",
        pct=False,
        axis=0,
    ):
        """
        Return the rank of values within each group.
        """
        if not axis == 0:
            raise NotImplementedError("Only axis=0 is supported.")

        if na_option not in {"keep", "top", "bottom"}:
            raise ValueError(
                f"na_option must be one of 'keep', 'top', or 'bottom', "
                f"but got {na_option}"
            )

        # TODO: in pandas compatibility mode, we should convert any
        # NaNs to nulls in any float value columns, as Pandas
        # treats NaNs the way we treat nulls.
        if cudf.get_option("mode.pandas_compatible"):
            if any(
                col.dtype.kind == "f" for col in self.grouping.values._columns
            ):
                raise NotImplementedError(
                    "NaNs are not supported in groupby.rank."
                )

        def rank(x):
            return getattr(x, "rank")(
                method=method,
                ascending=ascending,
                na_option=na_option,
                pct=pct,
            )

        result = self.agg(rank)

        if cudf.get_option("mode.pandas_compatible"):
            # pandas always returns floats:
            return result.astype("float64")

        return result

    @cached_property
    def _groupby(self):
        return libgroupby.GroupBy(
            [*self.grouping.keys._columns], dropna=self._dropna
        )

    @_performance_tracking
    def agg(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):
        """
        Apply aggregation(s) to the groups.

        Parameters
        ----------
        func : str, callable, list or dict
            Argument specifying the aggregation(s) to perform on the
            groups. `func` can be any of the following:

              - string: the name of a supported aggregation
              - callable: a function that accepts a Series/DataFrame and
                performs a supported operation on it.
              - list: a list of strings/callables specifying the
                aggregations to perform on every column.
              - dict: a mapping of column names to string/callable
                specifying the aggregations to perform on those
                columns.

        See :ref:`the user guide <basics.groupby>` for supported
        aggregations.

        Returns
        -------
        A Series or DataFrame containing the combined results of the
        aggregation(s).

        Examples
        --------
        >>> import cudf
        >>> a = cudf.DataFrame({
        ...     'a': [1, 1, 2],
        ...     'b': [1, 2, 3],
        ...     'c': [2, 2, 1]
        ... })
        >>> a.groupby('a', sort=True).agg('sum')
           b  c
        a
        1  3  4
        2  3  1

        Specifying a list of aggregations to perform on each column.

        >>> import cudf
        >>> a = cudf.DataFrame({
        ...     'a': [1, 1, 2],
        ...     'b': [1, 2, 3],
        ...     'c': [2, 2, 1]
        ... })
        >>> a.groupby('a', sort=True).agg(['sum', 'min'])
            b       c
          sum min sum min
        a
        1   3   1   4   2
        2   3   3   1   1

        Using a dict to specify aggregations to perform per column.

        >>> import cudf
        >>> a = cudf.DataFrame({
        ...     'a': [1, 1, 2],
        ...     'b': [1, 2, 3],
        ...     'c': [2, 2, 1]
        ... })
        >>> a.groupby('a', sort=True).agg({'a': 'max', 'b': ['min', 'mean']})
            a   b
          max min mean
        a
        1   1   1  1.5
        2   2   3  3.0

        Using lambdas/callables to specify aggregations taking parameters.

        >>> import cudf
        >>> a = cudf.DataFrame({
        ...     'a': [1, 1, 2],
        ...     'b': [1, 2, 3],
        ...     'c': [2, 2, 1]
        ... })
        >>> f1 = lambda x: x.quantile(0.5); f1.__name__ = "q0.5"
        >>> f2 = lambda x: x.quantile(0.75); f2.__name__ = "q0.75"
        >>> a.groupby('a').agg([f1, f2])
             b          c
          q0.5 q0.75 q0.5 q0.75
        a
        1  1.5  1.75  2.0   2.0
        2  3.0  3.00  1.0   1.0
        """
        if engine is not None:
            raise NotImplementedError(
                "engine is non-functional and added for compatibility with pandas"
            )
        if engine_kwargs is not None:
            raise NotImplementedError(
                "engine_kwargs is non-functional added for compatibility with pandas"
            )
        if args:
            raise NotImplementedError(
                "Passing args to func is currently not supported."
            )

        column_names, columns, normalized_aggs = self._normalize_aggs(
            func, **kwargs
        )
        orig_dtypes = tuple(c.dtype for c in columns)

        # Note: When there are no key columns, the below produces
        # an Index with float64 dtype, while Pandas returns
        # an Index with int64 dtype.
        # (GH: 6945)
        (
            result_columns,
            grouped_key_cols,
            included_aggregations,
        ) = self._groupby.aggregate(columns, normalized_aggs)

        result_index = self.grouping.keys._from_columns_like_self(
            grouped_key_cols,
        )

        multilevel = _is_multi_agg(func)
        data = {}
        for col_name, aggs, cols, orig_dtype in zip(
            column_names,
            included_aggregations,
            result_columns,
            orig_dtypes,
        ):
            for agg_tuple, col in zip(aggs, cols):
                agg, agg_kind = agg_tuple
                agg_name = agg.__name__ if callable(agg) else agg
                if multilevel:
                    key = (col_name, agg_name)
                else:
                    key = col_name
                if (
                    agg in {list, "collect"}
                    and orig_dtype != col.dtype.element_type
                ):
                    # Structs lose their labels which we reconstruct here
                    col = col._with_type_metadata(cudf.ListDtype(orig_dtype))

                if agg_kind in {"COUNT", "SIZE", "ARGMIN", "ARGMAX"}:
                    data[key] = col.astype("int64")
                elif (
                    self.obj.empty
                    and (
                        isinstance(agg_name, str)
                        and agg_name in Reducible._SUPPORTED_REDUCTIONS
                    )
                    and len(col) == 0
                    and not isinstance(
                        col,
                        (
                            cudf.core.column.ListColumn,
                            cudf.core.column.StructColumn,
                            cudf.core.column.DecimalBaseColumn,
                        ),
                    )
                ):
                    data[key] = col.astype(orig_dtype)
                else:
                    data[key] = col
        data = ColumnAccessor(data, multiindex=multilevel)
        if not multilevel:
            data = data.rename_levels({np.nan: None}, level=0)
        result = cudf.DataFrame._from_data(data, index=result_index)

        if self._sort:
            result = result.sort_index()
        else:
            if cudf.get_option(
                "mode.pandas_compatible"
            ) and not libgroupby._is_all_scan_aggregate(normalized_aggs):
                # Even with `sort=False`, pandas guarantees that
                # groupby preserves the order of rows within each group.
                left_cols = list(self.grouping.keys.drop_duplicates()._columns)
                right_cols = list(result_index._columns)
                join_keys = [
                    _match_join_keys(lcol, rcol, "inner")
                    for lcol, rcol in zip(left_cols, right_cols)
                ]
                # TODO: In future, see if we can centralize
                # logic else where that has similar patterns.
                join_keys = map(list, zip(*join_keys))
                # By construction, left and right keys are related by
                # a permutation, so we can use an inner join.
                with acquire_spill_lock():
                    plc_tables = [
                        plc.Table(
                            [col.to_pylibcudf(mode="read") for col in cols]
                        )
                        for cols in join_keys
                    ]
                    left_plc, right_plc = plc.join.inner_join(
                        plc_tables[0],
                        plc_tables[1],
                        plc.types.NullEquality.EQUAL,
                    )
                    left_order = libcudf.column.Column.from_pylibcudf(left_plc)
                    right_order = libcudf.column.Column.from_pylibcudf(
                        right_plc
                    )
                # left order is some permutation of the ordering we
                # want, and right order is a matching gather map for
                # the result table. Get the correct order by sorting
                # the right gather map.
                (right_order,) = libcudf.sort.sort_by_key(
                    [right_order],
                    [left_order],
                    [True],
                    ["first"],
                    stable=False,
                )
                result = result._gather(
                    GatherMap.from_column_unchecked(
                        right_order, len(result), nullify=False
                    )
                )

        if not self._as_index:
            result = result.reset_index()
        if libgroupby._is_all_scan_aggregate(normalized_aggs):
            # Scan aggregations return rows in original index order
            return self._mimic_pandas_order(result)

        return result

    def _reduce_numeric_only(self, op: str):
        raise NotImplementedError(
            f"numeric_only is not implemented for {type(self)}"
        )

    def _reduce(
        self,
        op: str,
        numeric_only: bool = False,
        min_count: int = 0,
        *args,
        **kwargs,
    ):
        """Compute {op} of group values.

        Parameters
        ----------
        numeric_only : bool, default None
            Include only float, int, boolean columns. If None, will attempt to
            use everything, then use only numeric data.
        min_count : int, default 0
            The required number of valid values to perform the operation. If
            fewer than ``min_count`` non-NA values are present the result will
            be NA.

        Returns
        -------
        Series or DataFrame
            Computed {op} of values within each group.

        .. pandas-compat::
            :meth:`pandas.core.groupby.DataFrameGroupBy.{op}`,
             :meth:`pandas.core.groupby.SeriesGroupBy.{op}`

            The numeric_only, min_count
        """
        if min_count != 0:
            raise NotImplementedError(
                "min_count parameter is not implemented yet"
            )
        if numeric_only:
            return self._reduce_numeric_only(op)
        return self.agg(op)

    def _scan(self, op: str, *args, **kwargs):
        """{op_name} for each group."""
        return self.agg(op)

    aggregate = agg

    def _head_tail(self, n, *, take_head: bool, preserve_order: bool):
        """Return the head or tail of each group

        Parameters
        ----------
        n
           Number of entries to include (if negative, number of
           entries to exclude)
        take_head
           Do we want the head or the tail of the group
        preserve_order
            If True, return the n rows from each group in original
            dataframe order (this mimics pandas behavior though is
            more expensive).

        Returns
        -------
        New DataFrame or Series

        Notes
        -----
        Unlike pandas, this returns an object in group order, not
        original order, unless ``preserve_order`` is ``True``.
        """
        # A more memory-efficient implementation would merge the take
        # into the grouping, but that probably requires a new
        # aggregation scheme in libcudf. This is probably "fast
        # enough" for most reasonable input sizes.
        _, offsets, _, group_values = self._grouped()
        group_offsets = np.asarray(offsets, dtype=size_type_dtype)
        size_per_group = np.diff(group_offsets)
        # "Out of bounds" n for the group size either means no entries
        # (negative) or all the entries (positive)
        if n < 0:
            size_per_group = np.maximum(
                size_per_group + n, 0, out=size_per_group
            )
        else:
            size_per_group = np.minimum(size_per_group, n, out=size_per_group)
        if take_head:
            group_offsets = group_offsets[:-1]
        else:
            group_offsets = group_offsets[1:] - size_per_group
        to_take = np.arange(size_per_group.sum(), dtype=size_type_dtype)
        fixup = np.empty_like(size_per_group)
        fixup[0] = 0
        np.cumsum(size_per_group[:-1], out=fixup[1:])
        to_take += np.repeat(group_offsets - fixup, size_per_group)
        to_take = as_column(to_take)
        result = group_values.iloc[to_take]
        if preserve_order:
            # Can't use _mimic_pandas_order because we need to
            # subsample the gather map from the full input ordering,
            # rather than permuting the gather map of the output.
            _, _, (ordering,) = self._groupby.groups(
                [as_column(range(0, len(self.obj)))]
            )
            # Invert permutation from original order to groups on the
            # subset of entries we want.
            gather_map = ordering.take(to_take).argsort()
            return result.take(gather_map)
        else:
            return result

    @_performance_tracking
    def head(self, n: int = 5, *, preserve_order: bool = True):
        """Return first n rows of each group

        Parameters
        ----------
        n
            If positive: number of entries to include from start of group
            If negative: number of entries to exclude from end of group

        preserve_order
            If True (default), return the n rows from each group in
            original dataframe order (this mimics pandas behavior
            though is more expensive). If you don't need rows in
            original dataframe order you will see a performance
            improvement by setting ``preserve_order=False``. In both
            cases, the original index is preserved, so ``.loc``-based
            indexing will work identically.

        Returns
        -------
        Series or DataFrame
            Subset of the original grouped object as determined by n

        See Also
        --------
        .tail

        Examples
        --------
        >>> df = cudf.DataFrame(
        ...     {
        ...         "a": [1, 0, 1, 2, 2, 1, 3, 2, 3, 3, 3],
        ...         "b": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ...     }
        ... )
        >>> df.groupby("a").head(1)
           a  b
        0  1  0
        1  0  1
        3  2  3
        6  3  6
        >>> df.groupby("a").head(-2)
           a  b
        0  1  0
        3  2  3
        6  3  6
        8  3  8
        """
        return self._head_tail(
            n, take_head=True, preserve_order=preserve_order
        )

    @_performance_tracking
    def tail(self, n: int = 5, *, preserve_order: bool = True):
        """Return last n rows of each group

        Parameters
        ----------
        n
            If positive: number of entries to include from end of group
            If negative: number of entries to exclude from start of group

        preserve_order
            If True (default), return the n rows from each group in
            original dataframe order (this mimics pandas behavior
            though is more expensive). If you don't need rows in
            original dataframe order you will see a performance
            improvement by setting ``preserve_order=False``. In both
            cases, the original index is preserved, so ``.loc``-based
            indexing will work identically.

        Returns
        -------
        Series or DataFrame
            Subset of the original grouped object as determined by n


        See Also
        --------
        .head

        Examples
        --------
        >>> df = cudf.DataFrame(
        ...     {
        ...         "a": [1, 0, 1, 2, 2, 1, 3, 2, 3, 3, 3],
        ...         "b": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ...     }
        ... )
        >>> df.groupby("a").tail(1)
            a   b
        1   0   1
        5   1   5
        7   2   7
        10  3  10
        >>> df.groupby("a").tail(-2)
            a   b
        5   1   5
        7   2   7
        9   3   9
        10  3  10
        """
        return self._head_tail(
            n, take_head=False, preserve_order=preserve_order
        )

    @_performance_tracking
    def nth(self, n, dropna: Literal["any", "all", None] = None):
        """
        Return the nth row from each group.
        """
        if dropna is not None:
            raise NotImplementedError("dropna is not currently supported.")
        self.obj["__groupbynth_order__"] = range(0, len(self.obj))  # type: ignore[index]
        # We perform another groupby here to have the grouping columns
        # be a part of dataframe columns.
        result = self.obj.groupby(self.grouping.keys).agg(lambda x: x.nth(n))
        sizes = self.size().reindex(result.index)

        result = result[sizes > n]

        result.index = self.obj.index.take(
            result._data["__groupbynth_order__"]
        )
        del result._data["__groupbynth_order__"]
        del self.obj._data["__groupbynth_order__"]
        return result

    @_performance_tracking
    def ngroup(self, ascending=True):
        """
        Number each group from 0 to the number of groups - 1.

        This is the enumerative complement of cumcount. Note that the
        numbers given to the groups match the order in which the groups
        would be seen when iterating over the groupby object, not the
        order they are first observed.

        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from number of group - 1 to 0.

        Returns
        -------
        Series
            Unique numbers for each group.

        See Also
        --------
        .cumcount : Number the rows in each group.

        Examples
        --------
        >>> df = cudf.DataFrame({"A": list("aaabba")})
        >>> df
           A
        0  a
        1  a
        2  a
        3  b
        4  b
        5  a
        >>> df.groupby('A').ngroup()
        0    0
        1    0
        2    0
        3    1
        4    1
        5    0
        dtype: int64
        >>> df.groupby('A').ngroup(ascending=False)
        0    1
        1    1
        2    1
        3    0
        4    0
        5    1
        dtype: int64
        >>> df.groupby(["A", [1,1,2,3,2,1]]).ngroup()
        0    0
        1    0
        2    1
        3    3
        4    2
        5    0
        dtype: int64
        """
        index = self.grouping.keys.unique().sort_values()
        num_groups = len(index)
        has_null_group = any(col.has_nulls() for col in index._columns)
        if ascending:
            # Count ascending from 0 to num_groups - 1
            groups = range(num_groups)
        elif has_null_group:
            # Count descending from num_groups - 1 to 0, but subtract one more
            # for the null group making it num_groups - 2 to -1.
            groups = range(num_groups - 2, -2, -1)
        else:
            # Count descending from num_groups - 1 to 0
            groups = range(num_groups - 1, -1, -1)

        group_ids = cudf.Series._from_column(as_column(groups))

        if has_null_group:
            group_ids.iloc[-1] = cudf.NA

        group_ids.index = index
        return self._broadcast(group_ids)

    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        replace: bool = False,
        weights: abc.Sequence | "cudf.Series" | None = None,
        random_state: np.random.RandomState | int | None = None,
    ):
        """Return a random sample of items in each group.

        Parameters
        ----------
        n
            Number of items to return for each group, if sampling
            without replacement must be at most the size of the
            smallest group. Cannot be used with frac. Default is
            ``n=1`` if frac is None.
        frac
            Fraction of items to return. Cannot be used with n.
        replace
            Should sampling occur with or without replacement?
        weights
            Sampling probability for each element. Must be the same
            length as the grouped frame. Not currently supported.
        random_state
            Seed for random number generation.

        Returns
        -------
        New dataframe or series with samples of appropriate size drawn
        from each group.

        """
        if weights is not None:
            # To implement this case again needs different algorithms
            # in both cases.
            #
            # Without replacement, use the weighted reservoir sampling
            # approach of Efraimidas and Spirakis (2006)
            # https://doi.org/10.1016/j.ipl.2005.11.003, essentially,
            # do a segmented argsort sorting on weight-scaled
            # logarithmic deviates. See
            # https://timvieira.github.io/blog/post/
            # 2019/09/16/algorithms-for-sampling-without-replacement/
            #
            # With replacement is trickier, one might be able to use
            # the alias method, otherwise we're back to bucketed
            # rejection sampling.
            raise NotImplementedError("Sampling with weights is not supported")
        if frac is not None and n is not None:
            raise ValueError("Cannot supply both of frac and n")
        elif n is None and frac is None:
            n = 1
        elif frac is not None and not (0 <= frac <= 1):
            raise ValueError(
                "Sampling with fraction must provide fraction in "
                f"[0, 1], got {frac=}"
            )
        # TODO: handle random states properly.
        if random_state is not None and not isinstance(random_state, int):
            raise NotImplementedError(
                "Only integer seeds are supported for random_state "
                "in this case"
            )
        # Get the groups
        # TODO: convince Cython to convert the std::vector offsets
        # into a numpy array directly, rather than a list.
        # TODO: this uses the sort-based groupby, could one use hash-based?
        _, offsets, _, group_values = self._grouped()
        group_offsets = np.asarray(offsets, dtype=size_type_dtype)
        size_per_group = np.diff(group_offsets)
        if n is not None:
            samples_per_group = np.broadcast_to(
                size_type_dtype.type(n), size_per_group.shape
            )
            if not replace and (minsize := size_per_group.min()) < n:
                raise ValueError(
                    f"Cannot sample {n=} without replacement, "
                    f"smallest group is {minsize}"
                )
        else:
            # Pandas uses round-to-nearest, ties to even to
            # pick sample sizes for the fractional case (unlike IEEE
            # which is round-to-nearest, ties to sgn(x) * inf).
            samples_per_group = np.round(
                size_per_group * frac, decimals=0
            ).astype(size_type_dtype)
        if replace:
            # We would prefer to use cupy here, but their rng.integers
            # interface doesn't take array-based low and high
            # arguments.
            low = 0
            high = np.repeat(size_per_group, samples_per_group)
            rng = np.random.default_rng(seed=random_state)
            indices = rng.integers(low, high, dtype=size_type_dtype)
            indices += np.repeat(group_offsets[:-1], samples_per_group)
        else:
            # Approach: do a segmented argsort of the index array and take
            # the first samples_per_group entries from sorted array.
            # We will shuffle the group indices and then pick them out
            # from the grouped dataframe index.
            nrows = len(group_values)
            indices = cp.arange(nrows, dtype=size_type_dtype)
            if len(size_per_group) < 500:
                # Empirically shuffling with cupy is faster at this scale
                rs = cp.random.get_random_state()
                rs.seed(seed=random_state)
                for off, size in zip(group_offsets, size_per_group):
                    rs.shuffle(indices[off : off + size])
            else:
                rng = cp.random.default_rng(seed=random_state)
                (indices,) = segmented_sort_by_key(
                    [as_column(indices)],
                    [as_column(rng.random(size=nrows))],
                    as_column(group_offsets),
                    [],
                    [],
                    stable=True,
                )
                indices = cp.asarray(indices.data_array_view(mode="read"))
            # Which indices are we going to want?
            want = np.arange(samples_per_group.sum(), dtype=size_type_dtype)
            scan = np.empty_like(samples_per_group)
            scan[0] = 0
            np.cumsum(samples_per_group[:-1], out=scan[1:])
            want += np.repeat(group_offsets[:-1] - scan, samples_per_group)
            indices = indices[want]
        return group_values.iloc[indices]

    def serialize(self):
        header = {}
        frames = []

        header["kwargs"] = {
            "sort": self._sort,
            "dropna": self._dropna,
            "as_index": self._as_index,
        }

        obj_header, obj_frames = self.obj.serialize()
        header["obj"] = obj_header
        header["obj_type"] = pickle.dumps(type(self.obj))
        header["num_obj_frames"] = len(obj_frames)
        frames.extend(obj_frames)

        grouping_header, grouping_frames = self.grouping.serialize()
        header["grouping"] = grouping_header
        header["num_grouping_frames"] = len(grouping_frames)
        frames.extend(grouping_frames)

        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        kwargs = header["kwargs"]

        obj_type = pickle.loads(header["obj_type"])
        obj = obj_type.deserialize(
            header["obj"], frames[: header["num_obj_frames"]]
        )
        grouping = _Grouping.deserialize(
            header["grouping"], frames[header["num_obj_frames"] :]
        )
        return cls(obj, grouping, **kwargs)

    def _grouped(self, *, include_groups: bool = True):
        offsets, grouped_key_cols, grouped_value_cols = self._groupby.groups(
            [*self.obj.index._columns, *self.obj._columns]
        )
        grouped_keys = cudf.core.index._index_from_data(
            dict(enumerate(grouped_key_cols))
        )
        if isinstance(self.grouping.keys, cudf.MultiIndex):
            grouped_keys.names = self.grouping.keys.names
            to_drop = self.grouping.keys.names
        else:
            grouped_keys.name = self.grouping.keys.name
            to_drop = (self.grouping.keys.name,)
        grouped_values = self.obj._from_columns_like_self(
            grouped_value_cols,
            column_names=self.obj._column_names,
            index_names=self.obj._index_names,
        )
        if not include_groups:
            for col_name in to_drop:
                del grouped_values[col_name]
        group_names = grouped_keys.unique().sort_values()
        return (group_names, offsets, grouped_keys, grouped_values)

    def _normalize_aggs(
        self, aggs: MultiColumnAggType, **kwargs
    ) -> tuple[Iterable[Any], tuple[ColumnBase, ...], list[list[AggType]]]:
        """
        Normalize aggs to a list of list of aggregations, where `out[i]`
        is a list of aggregations for column `self.obj[i]`. We support four
        different form of `aggs` input here:
        - A single agg, such as "sum". This agg is applied to all value
        columns.
        - A list of aggs, such as ["sum", "mean"]. All aggs are applied to all
        value columns.
        - A mapping of column name to aggs, such as
        {"a": ["sum"], "b": ["mean"]}, the aggs are applied to specified
        column.
        - Pairs of column name and agg tuples passed as kwargs
        eg. col1=("a", "sum"), col2=("b", "prod"). The output column names are
        the keys. The aggs are applied to the corresponding column in the tuple.
        Each agg can be string or lambda functions.
        """

        aggs_per_column: Iterable[AggType | Iterable[AggType]]
        # TODO: Remove isinstance condition when the legacy dask_cudf API is removed.
        # See https://github.com/rapidsai/cudf/pull/16528#discussion_r1715482302 for information.
        if aggs or isinstance(aggs, dict):
            if isinstance(aggs, dict):
                column_names, aggs_per_column = aggs.keys(), aggs.values()
                columns = tuple(self.obj._data[col] for col in column_names)
            else:
                values = self.grouping.values
                column_names = values._column_names
                columns = values._columns
                aggs_per_column = (aggs,) * len(columns)
        elif not aggs and kwargs:
            column_names = kwargs.keys()

            def _raise_invalid_type(x):
                raise TypeError(
                    f"Invalid keyword argument {x} of type {type(x)} was passed to agg"
                )

            columns, aggs_per_column = zip(
                *(
                    (self.obj._data[x[0]], x[1])
                    if isinstance(x, tuple)
                    else _raise_invalid_type(x)
                    for x in kwargs.values()
                )
            )
        else:
            raise TypeError("Must provide at least one aggregation function.")

        # is_list_like performs type narrowing but type-checkers don't
        # know it. One could add a TypeGuard annotation to
        # is_list_like (see PEP647), but that is less useful than it
        # seems because unlike the builtin narrowings it only performs
        # narrowing in the positive case.
        normalized_aggs = [
            list(agg) if is_list_like(agg) else [agg]  # type: ignore
            for agg in aggs_per_column
        ]
        return column_names, columns, normalized_aggs

    @_performance_tracking
    def pipe(self, func, *args, **kwargs):
        """
        Apply a function `func` with arguments to this GroupBy
        object and return the function's result.

        Parameters
        ----------
        func : function
            Function to apply to this GroupBy object or,
            alternatively, a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the GroupBy object.
        args : iterable, optional
            Positional arguments passed into ``func``.
        kwargs : mapping, optional
            A dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : the return type of ``func``.

        See Also
        --------
        cudf.Series.pipe
            Apply a function with arguments to a series.

        cudf.DataFrame.pipe
            Apply a function with arguments to a dataframe.

        apply
            Apply function to each group instead of to the full GroupBy object.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'A': ['a', 'b', 'a', 'b'], 'B': [1, 2, 3, 4]})
        >>> df
           A  B
        0  a  1
        1  b  2
        2  a  3
        3  b  4

        To get the difference between each groups maximum and minimum value
        in one pass, you can do

        >>> df.groupby('A', sort=True).pipe(lambda x: x.max() - x.min())
           B
        A
        a  2
        b  2
        """
        return cudf.core.common.pipe(self, func, *args, **kwargs)

    @_performance_tracking
    def _jit_groupby_apply(
        self, function, group_names, offsets, group_keys, grouped_values, *args
    ):
        chunk_results = jit_groupby_apply(
            offsets, grouped_values, function, *args
        )
        return self._post_process_chunk_results(
            chunk_results, group_names, group_keys, grouped_values
        )

    @_performance_tracking
    def _iterative_groupby_apply(
        self, function, group_names, offsets, group_keys, grouped_values, *args
    ):
        ngroups = len(offsets) - 1
        if ngroups > self._MAX_GROUPS_BEFORE_WARN:
            warnings.warn(
                f"GroupBy.apply() performance scales poorly with "
                f"number of groups. Got {ngroups} groups. Some functions "
                "may perform better by passing engine='jit'",
                RuntimeWarning,
            )

        chunks = [grouped_values[s:e] for s, e in itertools.pairwise(offsets)]
        chunk_results = [function(chk, *args) for chk in chunks]
        return self._post_process_chunk_results(
            chunk_results, group_names, group_keys, grouped_values
        )

    def _post_process_chunk_results(
        self, chunk_results, group_names, group_keys, grouped_values
    ):
        if not len(chunk_results):
            return self.obj.head(0)
        if isinstance(chunk_results, ColumnBase) or cudf.api.types.is_scalar(
            chunk_results[0]
        ):
            data = ColumnAccessor(
                {None: as_column(chunk_results)}, verify=False
            )
            ty = cudf.Series if self._as_index else cudf.DataFrame
            result = ty._from_data(data, index=group_names)
            result.index.names = self.grouping.names
            return result

        elif isinstance(chunk_results[0], cudf.Series) and isinstance(
            self.obj, cudf.DataFrame
        ):
            # When the UDF is like df.sum(), the result for each
            # group is a row-like "Series" where the index labels
            # are the same as the original calling DataFrame
            if _is_row_of(chunk_results[0], self.obj):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    result = cudf.concat(chunk_results, axis=1).T
                result.index = group_names
                result.index.names = self.grouping.names
            # When the UDF is like df.x + df.y, the result for each
            # group is the same length as the original group
            elif (total_rows := sum(len(chk) for chk in chunk_results)) in {
                len(self.obj),
                len(group_names),
            }:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    result = cudf.concat(chunk_results)
                if total_rows == len(group_names):
                    result.index = group_names
                    # TODO: Is there a better way to determine what
                    # the column name should be, especially if we applied
                    # a nameless UDF.
                    result = result.to_frame(
                        name=grouped_values._column_names[0]
                    )
                else:
                    index_data = group_keys._data.copy(deep=True)
                    index_data[None] = grouped_values.index._column
                    result.index = cudf.MultiIndex._from_data(index_data)
            elif len(chunk_results) == len(group_names):
                result = cudf.concat(chunk_results, axis=1).T
                result.index = group_names
                result.index.names = self.grouping.names
            else:
                raise TypeError(
                    "Error handling Groupby apply output with input of "
                    f"type {type(self.obj)} and output of "
                    f"type {type(chunk_results[0])}"
                )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                result = cudf.concat(chunk_results)
            if self._group_keys:
                index_data = group_keys._data.copy(deep=True)
                index_data[None] = grouped_values.index._column
                result.index = cudf.MultiIndex._from_data(index_data)
        return result

    @_performance_tracking
    def apply(
        self, func, *args, engine="auto", include_groups: bool = True, **kwargs
    ):
        """Apply a python transformation function over the grouped chunk.

        Parameters
        ----------
        func : callable
          The python transformation function that will be applied
          on the grouped chunk.
        args : tuple
            Optional positional arguments to pass to the function.
        engine: 'auto', 'cudf', or 'jit', default 'auto'
          Selects the GroupBy.apply implementation. Use `jit` to
          select the numba JIT pipeline. Only certain operations are allowed
          within the function when using this option: min, max, sum, mean, var,
          std, idxmax, and idxmin and any arithmetic formula involving them are
          allowed. Binary operations are not yet supported, so syntax like
          `df['x'] * 2` is not yet allowed.
          For more information, see the `cuDF guide to user defined functions
          <https://docs.rapids.ai/api/cudf/stable/user_guide/guide-to-udfs.html>`__.
          Use `cudf` to select the iterative groupby apply algorithm which aims
          to provide maximum flexibility at the expense of performance.
          The default value `auto` will attempt to use the numba JIT pipeline
          where possible and will fall back to the iterative algorithm if
          necessary.
        include_groups : bool, default True
            When True, will attempt to apply ``func`` to the groupings in
            the case that they are columns of the DataFrame. In the future,
            this will default to ``False``.
        kwargs : dict
            Optional keyword arguments to pass to the function.
            Currently not supported

        Examples
        --------
        .. code-block:: python

          from cudf import DataFrame
          df = DataFrame()
          df['key'] = [0, 0, 1, 1, 2, 2, 2]
          df['val'] = [0, 1, 2, 3, 4, 5, 6]
          groups = df.groupby(['key'])

          # Define a function to apply to each row in a group
          def mult(df):
            df['out'] = df['key'] * df['val']
            return df

          result = groups.apply(mult)
          print(result)

        Output:

        .. code-block:: python

             key  val  out
          0    0    0    0
          1    0    1    0
          2    1    2    2
          3    1    3    3
          4    2    4    8
          5    2    5   10
          6    2    6   12

        .. pandas-compat::
            :meth:`pandas.core.groupby.DataFrameGroupBy.apply`,
             :meth:`pandas.core.groupby.SeriesGroupBy.apply`

            cuDF's ``groupby.apply`` is limited compared to pandas.
            In some situations, Pandas returns the grouped keys as part of
            the index while cudf does not due to redundancy. For example:

            .. code-block::

                >>> import pandas as pd
                >>> df = pd.DataFrame({
                ...     'a': [1, 1, 2, 2],
                ...     'b': [1, 2, 1, 2],
                ...     'c': [1, 2, 3, 4],
                ... })
                >>> gdf = cudf.from_pandas(df)
                >>> df.groupby('a')[["b", "c"]].apply(lambda x: x.iloc[[0]])
                     b  c
                a
                1 0  1  1
                2 2  1  3
                >>> gdf.groupby('a')[["b", "c"]].apply(lambda x: x.iloc[[0]])
                   b  c
                0  1  1
                2  1  3

        ``engine='jit'`` may be used to accelerate certain functions,
        initially those that contain reductions and arithmetic operations
        between results of those reductions:

        >>> import cudf
        >>> df = cudf.DataFrame({'a':[1,1,2,2,3,3], 'b':[1,2,3,4,5,6]})
        >>> df.groupby('a').apply(
        ...   lambda group: group['b'].max() - group['b'].min(),
        ...   engine='jit'
        ... )
        a
        1    1
        2    1
        3    1
        dtype: int64

        """
        if kwargs:
            raise NotImplementedError(
                "Passing kwargs to func is currently not supported."
            )
        if self.obj.empty:
            if func in {"count", "size", "idxmin", "idxmax"}:
                res = cudf.Series([], dtype="int64")
            else:
                res = self.obj.copy(deep=True)
            res.index = self.grouping.keys
            if func in {"sum", "product"}:
                # For `sum` & `product`, boolean types
                # will need to result in `int64` type.
                for name, col in res._column_labels_and_values:
                    if col.dtype.kind == "b":
                        res._data[name] = col.astype("int")
            return res

        if not callable(func):
            raise TypeError(f"type {type(func)} is not callable")
        group_names, offsets, group_keys, grouped_values = self._grouped(
            include_groups=include_groups
        )

        if engine == "auto":
            if _can_be_jitted(grouped_values, func, args):
                engine = "jit"
            else:
                engine = "cudf"
        if engine == "jit":
            result = self._jit_groupby_apply(
                func,
                group_names,
                offsets,
                group_keys,
                grouped_values,
                *args,
            )
        elif engine == "cudf":
            result = self._iterative_groupby_apply(
                func,
                group_names,
                offsets,
                group_keys,
                grouped_values,
                *args,
            )
        else:
            raise ValueError(f"Unsupported engine '{engine}'")

        if self._sort:
            result = result.sort_index()
        if self._as_index is False:
            result = result.reset_index()
        return result

    @_performance_tracking
    def apply_grouped(self, function, **kwargs):
        """Apply a transformation function over the grouped chunk.

        This uses numba's CUDA JIT compiler to convert the Python
        transformation function into a CUDA kernel, thus will have a
        compilation overhead during the first run.

        Parameters
        ----------
        func : function
          The transformation function that will be executed on the CUDA GPU.
        incols: list
          A list of names of input columns.
        outcols: list
          A dictionary of output column names and their dtype.
        kwargs : dict
          name-value of extra arguments. These values are passed directly into
          the function.

        Examples
        --------
        .. code-block:: python

            from cudf import DataFrame
            from numba import cuda
            import numpy as np

            df = DataFrame()
            df['key'] = [0, 0, 1, 1, 2, 2, 2]
            df['val'] = [0, 1, 2, 3, 4, 5, 6]
            groups = df.groupby(['key'])

            # Define a function to apply to each group
            def mult_add(key, val, out1, out2):
                for i in range(cuda.threadIdx.x, len(key), cuda.blockDim.x):
                    out1[i] = key[i] * val[i]
                    out2[i] = key[i] + val[i]

            result = groups.apply_grouped(mult_add,
                                          incols=['key', 'val'],
                                          outcols={'out1': np.int32,
                                                   'out2': np.int32},
                                          # threads per block
                                          tpb=8)

            print(result)

        Output:

        .. code-block:: python

               key  val out1 out2
            0    0    0    0    0
            1    0    1    0    1
            2    1    2    2    3
            3    1    3    3    4
            4    2    4    8    6
            5    2    5   10    7
            6    2    6   12    8



        .. code-block:: python

            import cudf
            import numpy as np
            from numba import cuda
            import pandas as pd
            from random import randint


            # Create a random 15 row dataframe with one categorical
            # feature and one random integer valued feature
            df = cudf.DataFrame(
                    {
                        "cat": [1] * 5 + [2] * 5 + [3] * 5,
                        "val": [randint(0, 100) for _ in range(15)],
                    }
                 )

            # Group the dataframe by its categorical feature
            groups = df.groupby("cat")

            # Define a kernel which takes the moving average of a
            # sliding window
            def rolling_avg(val, avg):
                win_size = 3
                for i in range(cuda.threadIdx.x, len(val), cuda.blockDim.x):
                    if i < win_size - 1:
                        # If there is not enough data to fill the window,
                        # take the average to be NaN
                        avg[i] = np.nan
                    else:
                        total = 0
                        for j in range(i - win_size + 1, i + 1):
                            total += val[j]
                        avg[i] = total / win_size

            # Compute moving averages on all groups
            results = groups.apply_grouped(rolling_avg,
                                           incols=['val'],
                                           outcols=dict(avg=np.float64))
            print("Results:", results)

            # Note this gives the same result as its pandas equivalent
            pdf = df.to_pandas()
            pd_results = pdf.groupby('cat')['val'].rolling(3).mean()


        Output:

        .. code-block:: python

            Results:
               cat  val                 avg
            0    1   16
            1    1   45
            2    1   62                41.0
            3    1   45  50.666666666666664
            4    1   26  44.333333333333336
            5    2    5
            6    2   51
            7    2   77  44.333333333333336
            8    2    1                43.0
            9    2   46  41.333333333333336
            [5 more rows]

        This is functionally equivalent to `pandas.DataFrame.Rolling
        <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html>`_

        """
        if not callable(function):
            raise TypeError(f"type {type(function)} is not callable")

        _, offsets, _, grouped_values = self._grouped()
        kwargs.update({"chunks": offsets})
        return grouped_values.apply_chunks(function, **kwargs)

    @_performance_tracking
    def _broadcast(self, values: cudf.Series) -> cudf.Series:
        """
        Broadcast the results of an aggregation to the group

        Parameters
        ----------
        values: Series
            A Series representing the results of an aggregation.  The
            index of the Series must be the (unique) values
            representing the group keys.

        Returns
        -------
        A Series of the same size and with the same index as
        ``self.obj``.
        """
        if not values.index.equals(self.grouping.keys):
            values = values._align_to_index(
                self.grouping.keys, how="right", allow_non_unique=True
            )
            values.index = self.obj.index
        return values

    @_performance_tracking
    def transform(
        self, func, *args, engine=None, engine_kwargs=None, **kwargs
    ):
        """Apply an aggregation, then broadcast the result to the group size.

        Parameters
        ----------
        func: str or callable
            Aggregation to apply to each group. Note that the set of
            operations currently supported by `transform` is identical
            to that supported by the `agg` method.

        Returns
        -------
        A Series or DataFrame of the same size as the input, with the
        result of the aggregation per group broadcasted to the group
        size.

        Examples
        --------
        .. code-block:: python

          import cudf
          df = cudf.DataFrame({'a': [2, 1, 1, 2, 2], 'b': [1, 2, 3, 4, 5]})
          df.groupby('a').transform('max')
             b
          0  5
          1  3
          2  3
          3  5
          4  5

        See Also
        --------
        agg
        """
        if engine is not None:
            raise NotImplementedError(
                "engine is non-functional and added for compatibility with pandas"
            )
        if engine_kwargs is not None:
            raise NotImplementedError(
                "engine_kwargs is non-functional added for compatibility with pandas"
            )
        if args:
            raise NotImplementedError(
                "Passing args to func is currently not supported."
            )
        if kwargs:
            raise NotImplementedError(
                "Passing kwargs to func is currently not supported."
            )

        if not (isinstance(func, str) or callable(func)):
            raise TypeError(
                "Aggregation must be a named aggregation or a callable"
            )
        try:
            result = self.agg(func)
        except TypeError as e:
            raise NotImplementedError(
                "Currently, `transform()` supports only aggregations."
            ) from e
        # If the aggregation is a scan, don't broadcast
        if libgroupby._is_all_scan_aggregate([[func]]):
            if len(result) != len(self.obj):
                raise AssertionError(
                    "Unexpected result length for scan transform"
                )
            return result
        return self._broadcast(result)

    def rolling(self, *args, **kwargs):
        """
        Returns a `RollingGroupby` object that enables rolling window
        calculations on the groups.

        See Also
        --------
        cudf.core.window.Rolling
        """
        return cudf.core.window.rolling.RollingGroupby(self, *args, **kwargs)

    @_performance_tracking
    def count(self, dropna=True):
        """Compute the number of values in each column.

        Parameters
        ----------
        dropna : bool
            If ``True``, don't include null values in the count.
        """

        def func(x):
            return getattr(x, "count")(dropna=dropna)

        return self.agg(func)

    @_performance_tracking
    def describe(self, percentiles=None, include=None, exclude=None):
        """
        Generate descriptive statistics that summarizes the central tendency,
        dispersion and shape of a dataset's distribution, excluding NaN values.

        Analyzes numeric DataFrames only

        Parameters
        ----------
        percentiles : list-like of numbers, optional
            The percentiles to include in the output.
            Currently not supported.

        include: 'all', list-like of dtypes or None (default), optional
            list of data types to include in the result.
            Ignored for Series.

        exclude: list-like of dtypes or None (default), optional,
            list of data types to omit from the result.
            Ignored for Series.

        Returns
        -------
        Series or DataFrame
            Summary statistics of the Dataframe provided.

        Examples
        --------
        >>> import cudf
        >>> gdf = cudf.DataFrame({
        ...     "Speed": [380.0, 370.0, 24.0, 26.0],
        ...      "Score": [50, 30, 90, 80],
        ... })
        >>> gdf
           Speed  Score
        0  380.0     50
        1  370.0     30
        2   24.0     90
        3   26.0     80
        >>> gdf.groupby('Score').describe()
             Speed
             count   mean   std    min    25%    50%    75%     max
        Score
        30        1  370.0  <NA>  370.0  370.0  370.0  370.0  370.0
        50        1  380.0  <NA>  380.0  380.0  380.0  380.0  380.0
        80        1   26.0  <NA>   26.0   26.0   26.0   26.0   26.0
        90        1   24.0  <NA>   24.0   24.0   24.0   24.0   24.0

        """
        if percentiles is not None:
            raise NotImplementedError("percentiles is currently not supported")
        if exclude is not None:
            raise NotImplementedError("exclude is currently not supported")
        if include is not None:
            raise NotImplementedError("include is currently not supported")

        res = self.agg(
            [
                "count",
                "mean",
                "std",
                "min",
                _quantile_25,
                _quantile_50,
                _quantile_75,
                "max",
            ]
        )
        res.rename(
            columns={
                "_quantile_25": "25%",
                "_quantile_50": "50%",
                "_quantile_75": "75%",
            },
            level=1,
            inplace=True,
        )
        return res

    @_performance_tracking
    def cov(self, min_periods=0, ddof=1, numeric_only: bool = False):
        """
        Compute the pairwise covariance among the columns of a DataFrame,
        excluding NA/null values.

        The returned DataFrame is the covariance matrix of the columns of
        the DataFrame.

        Both NA and null values are automatically excluded from the
        calculation. See the note below about bias from missing values.

        A threshold can be set for the minimum number of observations
        for each value created. Comparisons with observations below this
        threshold will be returned as `NA`.

        This method is generally used for the analysis of time series data to
        understand the relationship between different measures across time.

        Parameters
        ----------
        min_periods: int, optional
            Minimum number of observations required per pair of columns
            to have a valid result.

        ddof: int, optional
            Delta degrees of freedom, default is 1.

        Returns
        -------
        DataFrame
            Covariance matrix.

        Notes
        -----
        Returns the covariance matrix of the DataFrame's time series.
        The covariance is normalized by N-ddof.

        For DataFrames that have Series that are missing data
        (assuming that data is missing at random) the returned covariance
        matrix will be an unbiased estimate of the variance and covariance
        between the member Series.

        However, for many applications this estimate may not be acceptable
        because the estimate covariance matrix is not guaranteed to be
        positive semi-definite. This could lead to estimate correlations
        having absolute values which are greater than one, and/or a
        non-invertible covariance matrix. See
        `Estimation of covariance matrices
        <https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices>`
        for more details.

        Examples
        --------
        >>> import cudf
        >>> gdf = cudf.DataFrame({
        ...     "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        ...     "val1": [5, 4, 6, 4, 8, 7, 4, 5, 2],
        ...     "val2": [4, 5, 6, 1, 2, 9, 8, 5, 1],
        ...     "val3": [4, 5, 6, 1, 2, 9, 8, 5, 1],
        ... })
        >>> gdf
          id  val1  val2  val3
        0  a     5     4     4
        1  a     4     5     5
        2  a     6     6     6
        3  b     4     1     1
        4  b     8     2     2
        5  b     7     9     9
        6  c     4     8     8
        7  c     5     5     5
        8  c     2     1     1
        >>> gdf.groupby("id").cov()
                    val1       val2       val3
        id
        a  val1  1.000000   0.500000   0.500000
           val2  0.500000   1.000000   1.000000
           val3  0.500000   1.000000   1.000000
        b  val1  4.333333   3.500000   3.500000
           val2  3.500000  19.000000  19.000000
           val3  3.500000  19.000000  19.000000
        c  val1  2.333333   3.833333   3.833333
           val2  3.833333  12.333333  12.333333
           val3  3.833333  12.333333  12.333333
        """
        if numeric_only is not False:
            raise NotImplementedError(
                "numeric_only is currently not supported."
            )

        return self._cov_or_corr(
            lambda x: x.cov(min_periods, ddof), "Covariance"
        )

    def _cov_or_corr(self, func, method_name):
        """
        Internal function that is called by either corr() or cov()
        for sort groupby correlation and covariance computations,
        respectively.
        """
        # create expanded dataframe consisting all combinations of the
        # struct columns-pairs to be used in the correlation or covariance
        # i.e. (('col1', 'col1'), ('col1', 'col2'), ('col2', 'col2'))
        column_names = self.grouping.values._column_names
        num_cols = len(column_names)

        column_pair_structs = {}
        for x, y in itertools.combinations_with_replacement(column_names, 2):
            # The number of output columns is the number of input columns
            # squared. We directly call the struct column factory here to
            # reduce overhead and avoid copying data. Since libcudf groupby
            # maintains a cache of aggregation requests, reusing the same
            # column also makes use of previously cached column means and
            # reduces kernel costs.

            # checks if input column names are string, raise a warning if
            # not so and cast them to strings
            if not (isinstance(x, str) and isinstance(y, str)):
                warnings.warn(
                    "DataFrame contains non-string column name(s). "
                    "Struct columns require field names to be strings. "
                    "Non-string column names will be cast to strings "
                    "in the result's field names."
                )
                x, y = str(x), str(y)

            column_pair_structs[(x, y)] = cudf.core.column.StructColumn(
                data=None,
                dtype=StructDtype(
                    fields={x: self.obj._data[x].dtype, y: self.obj._data[y]}
                ),
                children=(self.obj._data[x], self.obj._data[y]),
                size=len(self.obj),
                offset=0,
            )

        column_pair_groupby = cudf.DataFrame._from_data(
            column_pair_structs
        ).groupby(by=self.grouping.keys)

        try:
            gb_cov_corr = column_pair_groupby.agg(func)
        except RuntimeError as e:
            if "Unsupported groupby reduction type-agg combination" in str(e):
                raise TypeError(
                    f"{method_name} accepts only numerical column-pairs"
                )
            raise

        # ensure that column-pair labels are arranged in ascending order
        cols_list = [
            (y, x) if i > j else (x, y)
            for j, y in enumerate(column_names)
            for i, x in enumerate(column_names)
        ]
        cols_split = [
            cols_list[i : i + num_cols]
            for i in range(0, len(cols_list), num_cols)
        ]

        # interleave: combines the correlation or covariance results for each
        # column-pair into a single column

        @acquire_spill_lock()
        def interleave_columns(source_columns):
            return libcudf.column.Column.from_pylibcudf(
                plc.reshape.interleave_columns(
                    plc.Table(
                        [c.to_pylibcudf(mode="read") for c in source_columns]
                    )
                )
            )

        res = cudf.DataFrame._from_data(
            {
                x: interleave_columns([gb_cov_corr._data[y] for y in ys])
                for ys, x in zip(cols_split, column_names)
            }
        )

        # create a multiindex for the groupby covariance or correlation
        # dataframe, to match pandas behavior
        unsorted_idx = gb_cov_corr.index.repeat(num_cols)
        idx_sort_order = unsorted_idx._get_sorted_inds()
        sorted_idx = unsorted_idx._gather(idx_sort_order)
        if len(gb_cov_corr):
            # TO-DO: Should the operation below be done on the CPU instead?
            sorted_idx._data[None] = as_column(
                np.tile(column_names, len(gb_cov_corr.index))
            )
        res.index = MultiIndex._from_data(sorted_idx._data)

        return res

    @_performance_tracking
    def var(
        self,
        ddof=1,
        engine=None,
        engine_kwargs=None,
        numeric_only: bool = False,
    ):
        """Compute the column-wise variance of the values in each group.

        Parameters
        ----------
        ddof : int
            The delta degrees of freedom. N - ddof is the divisor used to
            normalize the variance.
        """
        if engine is not None:
            raise NotImplementedError(
                "engine is non-functional and added for compatibility with pandas"
            )
        if engine_kwargs is not None:
            raise NotImplementedError(
                "engine_kwargs is non-functional added for compatibility with pandas"
            )
        if numeric_only is not False:
            raise NotImplementedError(
                "numeric_only is currently not supported."
            )

        def func(x):
            return getattr(x, "var")(ddof=ddof)

        return self.agg(func)

    @_performance_tracking
    def nunique(self, dropna: bool = True):
        """
        Return number of unique elements in the group.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in the counts.
        """

        def func(x):
            return getattr(x, "nunique")(dropna=dropna)

        return self.agg(func)

    @_performance_tracking
    def std(
        self,
        ddof=1,
        engine=None,
        engine_kwargs=None,
        numeric_only: bool = False,
    ):
        """Compute the column-wise std of the values in each group.

        Parameters
        ----------
        ddof : int
            The delta degrees of freedom. N - ddof is the divisor used to
            normalize the standard deviation.
        """
        if engine is not None:
            raise NotImplementedError(
                "engine is non-functional and added for compatibility with pandas"
            )
        if engine_kwargs is not None:
            raise NotImplementedError(
                "engine_kwargs is non-functional added for compatibility with pandas"
            )
        if numeric_only is not False:
            raise NotImplementedError(
                "numeric_only is currently not supported."
            )

        def func(x):
            return getattr(x, "std")(ddof=ddof)

        return self.agg(func)

    @_performance_tracking
    def quantile(
        self, q=0.5, interpolation="linear", numeric_only: bool = False
    ):
        """Compute the column-wise quantiles of the values in each group.

        Parameters
        ----------
        q : float or array-like
            The quantiles to compute.
        interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}
            The interpolation method to use when the desired quantile lies
            between two data points. Defaults to "linear".
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.
            Currently not supported
        """
        if numeric_only is not False:
            raise NotImplementedError(
                "numeric_only is not currently supported."
            )

        def func(x):
            return getattr(x, "quantile")(q=q, interpolation=interpolation)

        return self.agg(func)

    @_performance_tracking
    def collect(self):
        """Get a list of all the values for each column in each group."""
        _deprecate_collect()
        return self.agg(list)

    @_performance_tracking
    def unique(self):
        """Get a list of the unique values for each column in each group."""
        return self.agg("unique")

    @_performance_tracking
    def diff(self, periods=1, axis=0):
        """Get the difference between the values in each group.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference,
            accepts negative values.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Take difference over rows (0) or columns (1).
            Only row-wise (0) shift is supported.

        Returns
        -------
        Series or DataFrame
            First differences of the Series or DataFrame.
        """

        if not axis == 0:
            raise NotImplementedError("Only axis=0 is supported.")

        values = self.obj.__class__._from_data(
            self.grouping.values._data, self.obj.index
        )
        return values - self.shift(periods=periods)

    def _scan_fill(self, method: str, limit: int) -> DataFrameOrSeries:
        """Internal implementation for `ffill` and `bfill`"""
        values = self.grouping.values
        result = self.obj._from_data(
            dict(
                zip(
                    values._column_names,
                    self._groupby.replace_nulls([*values._columns], method),
                )
            )
        )
        result = self._mimic_pandas_order(result)
        return result._copy_type_metadata(values)

    def ffill(self, limit=None):
        """Forward fill NA values.

        Parameters
        ----------
        limit : int, default None
            Unsupported
        """

        if limit is not None:
            raise NotImplementedError("Does not support limit param yet.")

        return self._scan_fill("ffill", limit)

    def bfill(self, limit=None):
        """Backward fill NA values.

        Parameters
        ----------
        limit : int, default None
            Unsupported
        """
        if limit is not None:
            raise NotImplementedError("Does not support limit param yet.")

        return self._scan_fill("bfill", limit)

    @_performance_tracking
    def fillna(
        self,
        value=None,
        method=None,
        axis=0,
        inplace=False,
        limit=None,
        downcast=None,
    ):
        """Fill NA values using the specified method.

        Parameters
        ----------
        value : scalar, dict
            Value to use to fill the holes. Cannot be specified with method.
        method : { 'bfill', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series

            - ffill: propagate last valid observation forward to next valid
            - bfill: use next valid observation to fill gap
        axis : {0 or 'index', 1 or 'columns'}
            Unsupported
        inplace : bool, default False
            If `True`, fill inplace. Note: this will modify other views on this
            object.
        limit : int, default None
            Unsupported
        downcast : dict, default None
            Unsupported

        Returns
        -------
        DataFrame or Series
        """
        warnings.warn(
            "groupby fillna is deprecated and "
            "will be removed in a future version. Use groupby ffill "
            "or groupby bfill for forward or backward filling instead.",
            FutureWarning,
        )
        if inplace:
            raise NotImplementedError("Does not support inplace yet.")
        if limit is not None:
            raise NotImplementedError("Does not support limit param yet.")
        if downcast is not None:
            raise NotImplementedError("Does not support downcast yet.")
        if not axis == 0:
            raise NotImplementedError("Only support axis == 0.")

        if value is None and method is None:
            raise ValueError("Must specify a fill 'value' or 'method'.")
        if value is not None and method is not None:
            raise ValueError("Cannot specify both 'value' and 'method'.")

        if method is not None:
            if method not in {"ffill", "bfill"}:
                raise ValueError("Method can only be of 'ffill', 'bfill'.")
            return getattr(self, method, limit)()

        values = self.obj.__class__._from_data(
            self.grouping.values._data, self.obj.index
        )
        return values.fillna(
            value=value, inplace=inplace, axis=axis, limit=limit
        )

    @_performance_tracking
    def shift(
        self,
        periods=1,
        freq=None,
        axis=0,
        fill_value=None,
        suffix: str | None = None,
    ):
        """
        Shift each group by ``periods`` positions.

        Parameters
        ----------
        periods : int, default 1
            Number of periods to shift.
        freq : str, unsupported
        axis : 0, axis to shift
            Shift direction. Only row-wise shift is supported
        fill_value : scalar or list of scalars, optional
            The scalar value to use for newly introduced missing values. Can be
            specified with `None`, a single value or multiple values:

            - `None` (default): sets all indeterminable values to null.
            - Single value: fill all shifted columns with this value. Should
              match the data type of all columns.
            - List of values: fill shifted columns with corresponding value in
              the list. The length of the list should match the number of
              columns shifted. Each value should match the data type of the
              column to fill.
        suffix : str, optional
            A string to add to each shifted column if there are multiple periods.
            Ignored otherwise.
            Currently not supported.

        Returns
        -------
        Series or DataFrame
            Object shifted within each group.

        .. pandas-compat::
            :meth:`pandas.core.groupby.DataFrameGroupBy.shift`,
             :meth:`pandas.core.groupby.SeriesGroupBy.shift`

            Parameter ``freq`` is unsupported.
        """

        if freq is not None:
            raise NotImplementedError("Parameter freq is unsupported.")

        if not axis == 0:
            raise NotImplementedError("Only axis=0 is supported.")

        if suffix is not None:
            raise NotImplementedError("shift is not currently supported.")

        values = self.grouping.values
        if is_list_like(fill_value):
            if len(fill_value) != len(values._data):
                raise ValueError(
                    "Mismatched number of columns and values to fill."
                )
        else:
            fill_value = [fill_value] * len(values._data)

        result = self.obj.__class__._from_data(
            dict(
                zip(
                    values._column_names,
                    self._groupby.shift(
                        [*values._columns], periods, fill_value
                    )[0],
                )
            )
        )
        result = self._mimic_pandas_order(result)
        return result._copy_type_metadata(values)

    @_performance_tracking
    def pct_change(
        self,
        periods=1,
        fill_method=no_default,
        axis=0,
        limit=no_default,
        freq=None,
    ):
        """
        Calculates the percent change between sequential elements
        in the group.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for forming percent change.
        fill_method : str, default 'ffill'
            How to handle NAs before computing percent changes.

            .. deprecated:: 24.04
                All options of `fill_method` are deprecated
                except `fill_method=None`.
        limit : int, optional
            The number of consecutive NAs to fill before stopping.
            Not yet implemented.

            .. deprecated:: 24.04
                `limit` is deprecated.
        freq : str, optional
            Increment to use from time series API.
            Not yet implemented.

        Returns
        -------
        Series or DataFrame
            Percentage changes within each group
        """
        if not axis == 0:
            raise NotImplementedError("Only axis=0 is supported.")
        if limit is not no_default:
            raise NotImplementedError("limit parameter not supported yet.")
        if freq is not None:
            raise NotImplementedError("freq parameter not supported yet.")
        elif fill_method not in {no_default, None, "ffill", "bfill"}:
            raise ValueError(
                "fill_method must be one of 'ffill', or" "'bfill'."
            )

        if fill_method not in (no_default, None) or limit is not no_default:
            # Do not remove until pandas 3.0 support is added.
            assert (
                PANDAS_LT_300
            ), "Need to drop after pandas-3.0 support is added."
            warnings.warn(
                "The 'fill_method' keyword being not None and the 'limit' "
                f"keywords in {type(self).__name__}.pct_change are "
                "deprecated and will be removed in a future version. "
                "Either fill in any non-leading NA values prior "
                "to calling pct_change or specify 'fill_method=None' "
                "to not fill NA values.",
                FutureWarning,
            )

        if fill_method in (no_default, None):
            fill_method = "ffill"
        if limit is no_default:
            limit = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filled = self.fillna(method=fill_method, limit=limit)

        fill_grp = filled.groupby(self.grouping)
        shifted = fill_grp.shift(periods=periods, freq=freq)
        return (filled / shifted) - 1

    def _mimic_pandas_order(
        self, result: DataFrameOrSeries
    ) -> DataFrameOrSeries:
        """Given a groupby result from libcudf, reconstruct the row orders
        matching that of pandas. This also adds appropriate indices.
        """
        # TODO: copy metadata after this method is a common pattern, should
        # merge in this method.

        # This function is used to reorder the results of scan-based
        # groupbys which have the same output size as input size.
        # However, if the grouping key has NAs and dropna=True, the
        # result coming back from libcudf has null_count few rows than
        # the input, so we must produce an ordering from the full
        # input range.
        _, _, (ordering,) = self._groupby.groups(
            [as_column(range(0, len(self.obj)))]
        )
        if self._dropna and any(
            c.has_nulls(include_nan=True) > 0
            for c in self.grouping._key_columns
        ):
            # Scan aggregations with null/nan keys put nulls in the
            # corresponding output rows in pandas, to do that here
            # expand the result by reindexing.
            ri = cudf.RangeIndex(0, len(self.obj))
            result.index = cudf.Index._from_column(ordering)
            # This reorders and expands
            result = result.reindex(ri)
        else:
            # Just reorder according to the groupings
            result = result.take(ordering.argsort())
        # Now produce the actual index we first thought of
        result.index = self.obj.index
        return result

    def ohlc(self):
        """
        Compute open, high, low and close values of a group, excluding missing values.

        Currently not implemented.
        """
        raise NotImplementedError("ohlc is currently not implemented")

    @property
    def plot(self):
        """
        Make plots of a grouped Series or DataFrame.

        Currently not implemented.
        """
        raise NotImplementedError("plot is currently not implemented")

    def resample(self, rule, *args, include_groups: bool = True, **kwargs):
        """
        Provide resampling when using a TimeGrouper.

        Currently not implemented.
        """
        raise NotImplementedError("resample is currently not implemented")

    def take(self, indices):
        """
        Return the elements in the given *positional* indices in each group.

        Currently not implemented.
        """
        raise NotImplementedError("take is currently not implemented")

    def filter(self, func, dropna: bool = True, *args, **kwargs):
        """
        Filter elements from groups that don't satisfy a criterion.

        Currently not implemented.
        """
        raise NotImplementedError("filter is currently not implemented")

    def expanding(self, *args, **kwargs):
        """
        Return an expanding grouper, providing expanding
        functionality per group.

        Currently not implemented.
        """
        raise NotImplementedError("expanding is currently not implemented")

    def ewm(self, *args, **kwargs):
        """
        Return an ewm grouper, providing ewm functionality per group.

        Currently not implemented.
        """
        raise NotImplementedError("expanding is currently not implemented")

    def any(self, skipna: bool = True):
        """
        Return True if any value in the group is truthful, else False.

        Currently not implemented.
        """
        raise NotImplementedError("any is currently not implemented")

    def all(self, skipna: bool = True):
        """
        Return True if all values in the group are truthful, else False.

        Currently not implemented.
        """
        raise NotImplementedError("all is currently not implemented")


class DataFrameGroupBy(GroupBy, GetAttrGetItemMixin):
    obj: "cudf.core.dataframe.DataFrame"

    _PROTECTED_KEYS = frozenset(("obj",))

    def _reduce_numeric_only(self, op: str):
        columns = list(
            name
            for name, dtype in self.obj._dtypes
            if (is_numeric_dtype(dtype) and name not in self.grouping.names)
        )
        return self[columns].agg(op)

    def __getitem__(self, key):
        return self.obj[key].groupby(
            by=self.grouping.keys,
            dropna=self._dropna,
            sort=self._sort,
            group_keys=self._group_keys,
            as_index=self._as_index,
        )

    def value_counts(
        self,
        subset=None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> DataFrameOrSeries:
        """
        Return a Series or DataFrame containing counts of unique rows.

        Parameters
        ----------
        subset : list-like, optional
            Columns to use when counting unique combinations.
        normalize : bool, default False
            Return proportions rather than frequencies.
        sort : bool, default True
            Sort by frequencies.
        ascending : bool, default False
            Sort in ascending order.
        dropna : bool, default True
            Don't include counts of rows that contain NA values.

        Returns
        -------
        Series or DataFrame
            Series if the groupby as_index is True, otherwise DataFrame.

        See Also
        --------
        Series.value_counts: Equivalent method on Series.
        DataFrame.value_counts: Equivalent method on DataFrame.
        SeriesGroupBy.value_counts: Equivalent method on SeriesGroupBy.

        Notes
        -----
        - If the groupby as_index is True then the returned Series will have a
          MultiIndex with one level per input column.
        - If the groupby as_index is False then the returned DataFrame will
          have an additional column with the value_counts. The column is
          labelled 'count' or 'proportion', depending on the ``normalize``
          parameter.

        By default, rows that contain any NA values are omitted from
        the result.

        By default, the result will be in descending order so that the
        first element of each group is the most frequently-occurring row.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({
        ...    'gender': ['male', 'male', 'female', 'male', 'female', 'male'],
        ...    'education': ['low', 'medium', 'high', 'low', 'high', 'low'],
        ...    'country': ['US', 'FR', 'US', 'FR', 'FR', 'FR']
        ... })

        >>> df
                gender  education   country
        0       male    low         US
        1       male    medium      FR
        2       female  high        US
        3       male    low         FR
        4       female  high        FR
        5       male    low         FR

        >>> df.groupby('gender').value_counts()
        gender  education  country
        female  high       FR         1
                           US         1
        male    low        FR         2
                           US         1
                medium     FR         1
        Name: count, dtype: int64

        >>> df.groupby('gender').value_counts(ascending=True)
        gender  education  country
        female  high       FR         1
                           US         1
        male    low        US         1
                medium     FR         1
                low        FR         2
        Name: count, dtype: int64

        >>> df.groupby('gender').value_counts(normalize=True)
        gender  education  country
        female  high       FR         0.50
                           US         0.50
        male    low        FR         0.50
                           US         0.25
                medium     FR         0.25
        Name: proportion, dtype: float64

        >>> df.groupby('gender', as_index=False).value_counts()
           gender education country  count
        0  female      high      FR      1
        1  female      high      US      1
        2    male       low      FR      2
        3    male       low      US      1
        4    male    medium      FR      1

        >>> df.groupby('gender', as_index=False).value_counts(normalize=True)
           gender education country  proportion
        0  female      high      FR        0.50
        1  female      high      US        0.50
        2    male       low      FR        0.50
        3    male       low      US        0.25
        4    male    medium      FR        0.25
        """

        df = cudf.DataFrame.copy(self.obj)
        groupings = self.grouping.names
        name = "proportion" if normalize else "count"

        if subset is None:
            subset = [i for i in df._column_names if i not in groupings]
        # Check subset exists in dataframe
        elif set(subset) - set(df._column_names):
            raise ValueError(
                f"Keys {set(subset) - set(df._column_names)} in subset "
                f"do not exist in the DataFrame."
            )
        # Catch case where groupby and subset share an element
        elif set(subset) & set(groupings):
            raise ValueError(
                f"Keys {set(subset) & set(groupings)} in subset "
                "cannot be in the groupby column keys."
            )

        df["__placeholder"] = 1
        result = (
            df.groupby(groupings + list(subset), dropna=dropna)[
                "__placeholder"
            ]
            .count()
            .sort_index()
            .astype(np.int64)
        )

        if normalize:
            levels = list(range(len(groupings), result.index.nlevels))
            result /= result.groupby(
                result.index.droplevel(levels),
            ).transform("sum")

        if sort:
            result = result.sort_values(ascending=ascending).sort_index(
                level=range(len(groupings)), sort_remaining=False
            )

        if not self._as_index:
            if name in df._column_names:
                raise ValueError(
                    f"Column label '{name}' is duplicate of result column"
                )
            result.name = name
            result = result.to_frame().reset_index()
        else:
            result.name = name

        return result

    @_performance_tracking
    def corr(
        self, method="pearson", min_periods=1, numeric_only: bool = False
    ):
        """
        Compute pairwise correlation of columns, excluding NA/null values.

        Parameters
        ----------
        method: {"pearson", "kendall", "spearman"} or callable,
            default "pearson". Currently only the pearson correlation
            coefficient is supported.

        min_periods: int, optional
            Minimum number of observations required per pair of columns
            to have a valid result.

        Returns
        -------
        DataFrame
            Correlation matrix.

        Examples
        --------
        >>> import cudf
        >>> gdf = cudf.DataFrame({
        ...             "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
        ...             "val1": [5, 4, 6, 4, 8, 7, 4, 5, 2],
        ...             "val2": [4, 5, 6, 1, 2, 9, 8, 5, 1],
        ...             "val3": [4, 5, 6, 1, 2, 9, 8, 5, 1]})
        >>> gdf
           id  val1  val2  val3
        0  a     5     4     4
        1  a     4     5     5
        2  a     6     6     6
        3  b     4     1     1
        4  b     8     2     2
        5  b     7     9     9
        6  c     4     8     8
        7  c     5     5     5
        8  c     2     1     1
        >>> gdf.groupby("id").corr(method="pearson")
                    val1      val2      val3
        id
        a   val1  1.000000  0.500000  0.500000
            val2  0.500000  1.000000  1.000000
            val3  0.500000  1.000000  1.000000
        b   val1  1.000000  0.385727  0.385727
            val2  0.385727  1.000000  1.000000
            val3  0.385727  1.000000  1.000000
        c   val1  1.000000  0.714575  0.714575
            val2  0.714575  1.000000  1.000000
            val3  0.714575  1.000000  1.000000
        """

        if method != "pearson":
            raise NotImplementedError(
                "Only pearson correlation is currently supported"
            )
        if numeric_only is not False:
            raise NotImplementedError(
                "numeric_only is currently not supported."
            )

        return self._cov_or_corr(
            lambda x: x.corr(method, min_periods), "Correlation"
        )

    def hist(
        self,
        column=None,
        by=None,
        grid: bool = True,
        xlabelsize: int | None = None,
        xrot: float | None = None,
        ylabelsize: int | None = None,
        yrot: float | None = None,
        ax=None,
        sharex: bool = False,
        sharey: bool = False,
        figsize: tuple[float, float] | None = None,
        layout: tuple[int, int] | None = None,
        bins: int | abc.Sequence[int] = 10,
        backend: str | None = None,
        legend: bool = False,
        **kwargs,
    ):
        raise NotImplementedError("hist is not currently implemented")

    def boxplot(
        self,
        subplots: bool = True,
        column=None,
        fontsize: int | None = None,
        rot: int = 0,
        grid: bool = True,
        ax=None,
        figsize: tuple[float, float] | None = None,
        layout=None,
        sharex: bool = False,
        sharey: bool = True,
        backend=None,
        **kwargs,
    ):
        raise NotImplementedError("boxplot is not currently implemented")


DataFrameGroupBy.__doc__ = groupby_doc_template.format(ret="")


class SeriesGroupBy(GroupBy):
    obj: "cudf.core.series.Series"

    def agg(self, func, *args, engine=None, engine_kwargs=None, **kwargs):
        result = super().agg(
            func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs
        )

        # downcast the result to a Series:
        if len(result._data):
            if result.shape[1] == 1 and not is_list_like(func):
                return result.iloc[:, 0]

        # drop the first level if we have a multiindex
        if result._data.nlevels > 1:
            result.columns = result._data.to_pandas_index().droplevel(0)

        return result

    aggregate = agg

    def apply(self, func, *args, **kwargs):
        result = super().apply(func, *args, **kwargs)

        # apply Series name to result
        result.name = self.obj.name

        return result

    @property
    def dtype(self) -> pd.Series:
        raise NotImplementedError("dtype is currently not implemented.")

    def hist(
        self,
        by=None,
        ax=None,
        grid: bool = True,
        xlabelsize: int | None = None,
        xrot: float | None = None,
        ylabelsize: int | None = None,
        yrot: float | None = None,
        figsize: tuple[float, float] | None = None,
        bins: int | abc.Sequence[int] = 10,
        backend: str | None = None,
        legend: bool = False,
        **kwargs,
    ):
        raise NotImplementedError("hist is currently not implemented.")

    @property
    def is_monotonic_increasing(self) -> cudf.Series:
        """
        Return whether each group's values are monotonically increasing.

        Currently not implemented
        """
        raise NotImplementedError(
            "is_monotonic_increasing is currently not implemented."
        )

    @property
    def is_monotonic_decreasing(self) -> cudf.Series:
        """
        Return whether each group's values are monotonically decreasing.

        Currently not implemented
        """
        raise NotImplementedError(
            "is_monotonic_decreasing is currently not implemented."
        )

    def nlargest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> cudf.Series:
        """
        Return the largest n elements.

        Currently not implemented
        """
        raise NotImplementedError("nlargest is currently not implemented.")

    def nsmallest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> cudf.Series:
        """
        Return the smallest n elements.

        Currently not implemented
        """
        raise NotImplementedError("nsmallest is currently not implemented.")

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ) -> cudf.Series | cudf.DataFrame:
        raise NotImplementedError("value_counts is currently not implemented.")

    def corr(
        self,
        other: cudf.Series,
        method: str = "pearson",
        min_periods: int | None = None,
    ) -> cudf.Series:
        raise NotImplementedError("corr is currently not implemented.")


SeriesGroupBy.__doc__ = groupby_doc_template.format(ret="")


# TODO: should we define this as a dataclass instead?
class Grouper:
    def __init__(
        self, key=None, level=None, freq=None, closed=None, label=None
    ):
        if key is not None and level is not None:
            raise ValueError("Grouper cannot specify both key and level")
        if (key, level) == (None, None) and not freq:
            raise ValueError("Grouper must specify either key or level")
        self.key = key
        self.level = level
        self.freq = freq
        self.closed = closed
        self.label = label


class _Grouping(Serializable):
    def __init__(self, obj, by=None, level=None):
        self._obj = obj
        self._key_columns = []
        self.names = []

        # Need to keep track of named key columns
        # to support `as_index=False` correctly
        self._named_columns = []
        self._handle_by_or_level(by, level)

        if len(obj) and not len(self._key_columns):
            raise ValueError("No group keys passed")

    def _handle_by_or_level(self, by=None, level=None):
        if level is not None:
            if by is not None:
                raise ValueError("Cannot specify both by and level")
            level_list = level if isinstance(level, list) else [level]
            for level in level_list:
                self._handle_level(level)
        else:
            by_list = by if isinstance(by, list) else [by]

            for by in by_list:
                if callable(by):
                    self._handle_callable(by)
                elif isinstance(by, cudf.Series):
                    self._handle_series(by)
                elif isinstance(by, cudf.BaseIndex):
                    self._handle_index(by)
                elif isinstance(by, abc.Mapping):
                    self._handle_mapping(by)
                elif isinstance(by, Grouper):
                    self._handle_grouper(by)
                elif isinstance(by, pd.Series):
                    self._handle_series(cudf.Series.from_pandas(by))
                elif isinstance(by, pd.Index):
                    self._handle_index(cudf.Index.from_pandas(by))
                else:
                    try:
                        self._handle_label(by)
                    except (KeyError, TypeError):
                        self._handle_misc(by)

    @property
    def keys(self):
        """Return grouping key columns as index"""
        nkeys = len(self._key_columns)

        if nkeys == 0:
            return cudf.Index([], name=None)
        elif nkeys > 1:
            return cudf.MultiIndex._from_data(
                dict(zip(range(nkeys), self._key_columns))
            )._set_names(self.names)
        else:
            return cudf.Index._from_column(
                self._key_columns[0], name=self.names[0]
            )

    @property
    def values(self) -> cudf.core.frame.Frame:
        """Return value columns as a frame.

        Note that in aggregation, value columns can be arbitrarily
        specified. While this method returns all non-key columns from `obj` as
        a frame.

        This is mainly used in transform-like operations.
        """
        # If the key columns are in `obj`, filter them out
        value_column_names = [
            x for x in self._obj._column_names if x not in self._named_columns
        ]
        value_columns = self._obj._data.select_by_label(value_column_names)
        return self._obj.__class__._from_data(value_columns)

    def _handle_callable(self, by):
        by = by(self._obj.index)
        self.__init__(self._obj, by)

    def _handle_series(self, by):
        by = by._align_to_index(self._obj.index, how="right")
        self._key_columns.append(by._column)
        self.names.append(by.name)

    def _handle_index(self, by):
        self._key_columns.extend(by._columns)
        self.names.extend(by._column_names)

    def _handle_mapping(self, by):
        by = cudf.Series(by.values(), index=by.keys())
        self._handle_series(by)

    def _handle_label(self, by):
        try:
            self._key_columns.append(self._obj._data[by])
        except KeyError as e:
            # `by` can be index name(label) too.
            if by in self._obj.index.names:
                self._key_columns.append(self._obj.index._data[by])
            else:
                raise e
        self.names.append(by)
        self._named_columns.append(by)

    def _handle_grouper(self, by):
        if by.freq:
            self._handle_frequency_grouper(by)
        elif by.key:
            self._handle_label(by.key)
        else:
            self._handle_level(by.level)

    def _handle_frequency_grouper(self, by):
        raise NotImplementedError()

    def _handle_level(self, by):
        level_values = self._obj.index.get_level_values(by)
        self._key_columns.append(level_values._values)
        self.names.append(level_values.name)

    def _handle_misc(self, by):
        by = cudf.core.column.as_column(by)
        if len(by) != len(self._obj):
            raise ValueError("Grouper and object must have same length")
        self._key_columns.append(by)
        self.names.append(None)

    def serialize(self):
        header = {}
        frames = []
        header["names"] = pickle.dumps(self.names)
        header["_named_columns"] = pickle.dumps(self._named_columns)
        column_header, column_frames = cudf.core.column.serialize_columns(
            self._key_columns
        )
        header["columns"] = column_header
        frames.extend(column_frames)
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        names = pickle.loads(header["names"])
        _named_columns = pickle.loads(header["_named_columns"])
        key_columns = cudf.core.column.deserialize_columns(
            header["columns"], frames
        )
        out = _Grouping.__new__(_Grouping)
        out.names = names
        out._named_columns = _named_columns
        out._key_columns = key_columns
        return out

    def copy(self, deep=True):
        out = _Grouping.__new__(_Grouping)
        out.names = copy.deepcopy(self.names)
        out._named_columns = copy.deepcopy(self._named_columns)
        out._key_columns = [col.copy(deep=deep) for col in self._key_columns]
        return out


def _is_multi_agg(aggs):
    """
    Returns True if more than one aggregation is performed
    on any of the columns as specified in `aggs`.
    """
    if isinstance(aggs, abc.Mapping):
        return any(is_list_like(agg) for agg in aggs.values())
    if is_list_like(aggs):
        return True
    return False
