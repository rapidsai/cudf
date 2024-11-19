# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from __future__ import annotations

from functools import wraps

import numpy as np
import pandas as pd

from dask.dataframe.core import (
    DataFrame as DaskDataFrame,
    aca,
    split_out_on_cols,
)
from dask.dataframe.groupby import DataFrameGroupBy, SeriesGroupBy
from dask.utils import funcname

import cudf
from cudf.core.groupby.groupby import _deprecate_collect
from cudf.utils.performance_tracking import _dask_cudf_performance_tracking

from dask_cudf._legacy.sorting import _deprecate_shuffle_kwarg

# aggregations that are dask-cudf optimized
OPTIMIZED_AGGS = (
    "count",
    "mean",
    "std",
    "var",
    "sum",
    "min",
    "max",
    list,
    "first",
    "last",
)


def _check_groupby_optimized(func):
    """
    Decorator for dask-cudf's groupby methods that returns the dask-cudf
    optimized method if the groupby object is supported, otherwise
    reverting to the upstream Dask method
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        gb = args[0]
        if _groupby_optimized(gb):
            return func(*args, **kwargs)
        # note that we use upstream Dask's default kwargs for this call if
        # none are specified; this shouldn't be an issue as those defaults are
        # consistent with dask-cudf
        return getattr(super(type(gb), gb), func.__name__)(*args[1:], **kwargs)

    return wrapper


class CudfDataFrameGroupBy(DataFrameGroupBy):
    @_dask_cudf_performance_tracking
    def __init__(self, *args, sort=None, **kwargs):
        self.sep = kwargs.pop("sep", "___")
        self.as_index = kwargs.pop("as_index", True)
        super().__init__(*args, sort=sort, **kwargs)

    @_dask_cudf_performance_tracking
    def __getitem__(self, key):
        if isinstance(key, list):
            g = CudfDataFrameGroupBy(
                self.obj,
                by=self.by,
                slice=key,
                sort=self.sort,
                **self.dropna,
            )
        else:
            g = CudfSeriesGroupBy(
                self.obj,
                by=self.by,
                slice=key,
                sort=self.sort,
                **self.dropna,
            )

        g._meta = g._meta[key]
        return g

    @_dask_cudf_performance_tracking
    def _make_groupby_method_aggs(self, agg_name):
        """Create aggs dictionary for aggregation methods"""

        if isinstance(self.by, list):
            return {c: agg_name for c in self.obj.columns if c not in self.by}
        return {c: agg_name for c in self.obj.columns if c != self.by}

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def count(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            self._make_groupby_method_aggs("count"),
            split_every,
            split_out,
        )

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def mean(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            self._make_groupby_method_aggs("mean"),
            split_every,
            split_out,
        )

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def std(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            self._make_groupby_method_aggs("std"),
            split_every,
            split_out,
        )

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def var(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            self._make_groupby_method_aggs("var"),
            split_every,
            split_out,
        )

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def sum(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            self._make_groupby_method_aggs("sum"),
            split_every,
            split_out,
        )

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def min(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            self._make_groupby_method_aggs("min"),
            split_every,
            split_out,
        )

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def max(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            self._make_groupby_method_aggs("max"),
            split_every,
            split_out,
        )

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def collect(self, split_every=None, split_out=1):
        _deprecate_collect()
        return _make_groupby_agg_call(
            self,
            self._make_groupby_method_aggs(list),
            split_every,
            split_out,
        )

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def first(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            self._make_groupby_method_aggs("first"),
            split_every,
            split_out,
        )

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def last(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            self._make_groupby_method_aggs("last"),
            split_every,
            split_out,
        )

    @_deprecate_shuffle_kwarg
    @_dask_cudf_performance_tracking
    def aggregate(
        self, arg, split_every=None, split_out=1, shuffle_method=None
    ):
        if arg == "size":
            return self.size()

        arg = _redirect_aggs(arg)

        if _groupby_optimized(self) and _aggs_optimized(arg, OPTIMIZED_AGGS):
            if isinstance(self._meta.grouping.keys, cudf.MultiIndex):
                keys = self._meta.grouping.keys.names
            else:
                keys = self._meta.grouping.keys.name

            return groupby_agg(
                self.obj,
                keys,
                arg,
                split_every=split_every,
                split_out=split_out,
                sep=self.sep,
                sort=self.sort,
                as_index=self.as_index,
                shuffle_method=shuffle_method,
                **self.dropna,
            )

        return super().aggregate(
            arg,
            split_every=split_every,
            split_out=split_out,
            shuffle_method=shuffle_method,
        )


class CudfSeriesGroupBy(SeriesGroupBy):
    @_dask_cudf_performance_tracking
    def __init__(self, *args, sort=None, **kwargs):
        self.sep = kwargs.pop("sep", "___")
        self.as_index = kwargs.pop("as_index", True)
        super().__init__(*args, sort=sort, **kwargs)

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def count(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            {self._slice: "count"},
            split_every,
            split_out,
        )[self._slice]

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def mean(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            {self._slice: "mean"},
            split_every,
            split_out,
        )[self._slice]

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def std(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            {self._slice: "std"},
            split_every,
            split_out,
        )[self._slice]

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def var(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            {self._slice: "var"},
            split_every,
            split_out,
        )[self._slice]

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def sum(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            {self._slice: "sum"},
            split_every,
            split_out,
        )[self._slice]

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def min(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            {self._slice: "min"},
            split_every,
            split_out,
        )[self._slice]

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def max(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            {self._slice: "max"},
            split_every,
            split_out,
        )[self._slice]

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def collect(self, split_every=None, split_out=1):
        _deprecate_collect()
        return _make_groupby_agg_call(
            self,
            {self._slice: list},
            split_every,
            split_out,
        )[self._slice]

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def first(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            {self._slice: "first"},
            split_every,
            split_out,
        )[self._slice]

    @_dask_cudf_performance_tracking
    @_check_groupby_optimized
    def last(self, split_every=None, split_out=1):
        return _make_groupby_agg_call(
            self,
            {self._slice: "last"},
            split_every,
            split_out,
        )[self._slice]

    @_deprecate_shuffle_kwarg
    @_dask_cudf_performance_tracking
    def aggregate(
        self, arg, split_every=None, split_out=1, shuffle_method=None
    ):
        if arg == "size":
            return self.size()

        arg = _redirect_aggs(arg)

        if not isinstance(arg, dict):
            arg = {self._slice: arg}

        if _groupby_optimized(self) and _aggs_optimized(arg, OPTIMIZED_AGGS):
            return _make_groupby_agg_call(
                self, arg, split_every, split_out, shuffle_method
            )[self._slice]

        return super().aggregate(
            arg,
            split_every=split_every,
            split_out=split_out,
            shuffle_method=shuffle_method,
        )


def _shuffle_aggregate(
    ddf,
    gb_cols,
    chunk,
    chunk_kwargs,
    aggregate,
    aggregate_kwargs,
    split_every,
    split_out,
    token=None,
    sort=None,
    shuffle_method=None,
):
    # Shuffle-based groupby aggregation
    # NOTE: This function is the dask_cudf version of
    # dask.dataframe.groupby._shuffle_aggregate

    # Step 1 - Chunkwise groupby operation
    chunk_name = f"{token or funcname(chunk)}-chunk"
    chunked = ddf.map_partitions(
        chunk,
        meta=chunk(ddf._meta, **chunk_kwargs),
        token=chunk_name,
        **chunk_kwargs,
    )

    # Step 2 - Perform global sort or shuffle
    shuffle_npartitions = max(
        chunked.npartitions // split_every,
        split_out,
    )
    if sort and split_out > 1:
        # Sort-based code path
        result = (
            chunked.repartition(npartitions=shuffle_npartitions)
            .sort_values(
                gb_cols,
                ignore_index=True,
                shuffle_method=shuffle_method,
            )
            .map_partitions(
                aggregate,
                meta=aggregate(chunked._meta, **aggregate_kwargs),
                **aggregate_kwargs,
            )
        )
    else:
        # Hash-based code path
        result = chunked.shuffle(
            gb_cols,
            npartitions=shuffle_npartitions,
            ignore_index=True,
            shuffle_method=shuffle_method,
        ).map_partitions(
            aggregate,
            meta=aggregate(chunked._meta, **aggregate_kwargs),
            **aggregate_kwargs,
        )

    # Step 3 - Repartition and return
    if split_out < result.npartitions:
        return result.repartition(npartitions=split_out)
    return result


@_dask_cudf_performance_tracking
def groupby_agg(
    ddf,
    gb_cols,
    aggs_in,
    split_every=None,
    split_out=None,
    dropna=True,
    sep="___",
    sort=False,
    as_index=True,
    shuffle_method=None,
):
    """Optimized groupby aggregation for Dask-CuDF.

    Parameters
    ----------
    ddf : DataFrame
        DataFrame object to perform grouping on.
    gb_cols : str or list[str]
        Column names to group by.
    aggs_in : str, list, or dict
        Aggregations to perform.
    split_every : int (optional)
        How to group intermediate aggregates.
    dropna : bool
        Drop grouping key values corresponding to NA values.
    as_index : bool
        Currently ignored.
    sort : bool
        Sort the group keys, better performance is obtained when
        not sorting.
    shuffle_method : str (optional)
        Control how shuffling of the DataFrame is performed.
    sep : str
        Internal usage.


    Notes
    -----
    This "optimized" approach is more performant than the algorithm in
    implemented in :meth:`DataFrame.apply` because it allows the cuDF
    backend to perform multiple aggregations at once.

    This aggregation algorithm only supports the following options

    * "list"
    * "count"
    * "first"
    * "last"
    * "max"
    * "mean"
    * "min"
    * "std"
    * "sum"
    * "var"


    See Also
    --------
    DataFrame.groupby : generic groupby of a DataFrame
    dask.dataframe.apply_concat_apply : for more description of the
        split_every argument.

    """
    # Assert that aggregations are supported
    aggs = _redirect_aggs(aggs_in)
    if not _aggs_optimized(aggs, OPTIMIZED_AGGS):
        raise ValueError(
            f"Supported aggs include {OPTIMIZED_AGGS} for groupby_agg API. "
            f"Aggregations must be specified with dict or list syntax."
        )

    # If split_every is False, we use an all-to-one reduction
    if split_every is False:
        split_every = max(ddf.npartitions, 2)

    # Deal with default split_out and split_every params
    split_every = split_every or 8
    split_out = split_out or 1

    # Standardize `gb_cols`, `columns`, and `aggs`
    if isinstance(gb_cols, str):
        gb_cols = [gb_cols]
    columns = [c for c in ddf.columns if c not in gb_cols]
    if not isinstance(aggs, dict):
        aggs = {col: aggs for col in columns}

    # Assert if our output will have a MultiIndex; this will be the case if
    # any value in the `aggs` dict is not a string (i.e. multiple/named
    # aggregations per column)
    str_cols_out = True
    aggs_renames = {}
    for col in aggs:
        if isinstance(aggs[col], str) or callable(aggs[col]):
            aggs[col] = [aggs[col]]
        elif isinstance(aggs[col], dict):
            str_cols_out = False
            col_aggs = []
            for k, v in aggs[col].items():
                aggs_renames[col, v] = k
                col_aggs.append(v)
            aggs[col] = col_aggs
        else:
            str_cols_out = False
        if col in gb_cols:
            columns.append(col)

    # Construct meta
    _aggs = aggs.copy()
    if str_cols_out:
        # Metadata should use `str` for dict values if that is
        # what the user originally specified (column names will
        # be str, rather than tuples).
        for col in aggs:
            _aggs[col] = _aggs[col][0]
    _meta = ddf._meta.groupby(gb_cols, as_index=as_index).agg(_aggs)
    if aggs_renames:
        col_array = []
        agg_array = []
        for col, agg in _meta.columns:
            col_array.append(col)
            agg_array.append(aggs_renames.get((col, agg), agg))
        _meta.columns = pd.MultiIndex.from_arrays([col_array, agg_array])

    chunk = _groupby_partition_agg
    chunk_kwargs = {
        "gb_cols": gb_cols,
        "aggs": aggs,
        "columns": columns,
        "dropna": dropna,
        "sort": sort,
        "sep": sep,
    }

    combine = _tree_node_agg
    combine_kwargs = {
        "gb_cols": gb_cols,
        "dropna": dropna,
        "sort": sort,
        "sep": sep,
    }

    aggregate = _finalize_gb_agg
    aggregate_kwargs = {
        "gb_cols": gb_cols,
        "aggs": aggs,
        "columns": columns,
        "final_columns": _meta.columns,
        "as_index": as_index,
        "dropna": dropna,
        "sort": sort,
        "sep": sep,
        "str_cols_out": str_cols_out,
        "aggs_renames": aggs_renames,
    }

    # Use shuffle_method=True for split_out>1
    if sort and split_out > 1 and shuffle_method is None:
        shuffle_method = "tasks"

    # Check if we are using the shuffle-based algorithm
    if shuffle_method:
        # Shuffle-based aggregation
        return _shuffle_aggregate(
            ddf,
            gb_cols,
            chunk,
            chunk_kwargs,
            aggregate,
            aggregate_kwargs,
            split_every,
            split_out,
            token="cudf-aggregate",
            sort=sort,
            shuffle_method=shuffle_method
            if isinstance(shuffle_method, str)
            else None,
        )

    # Deal with sort/shuffle defaults
    if split_out > 1 and sort:
        raise ValueError(
            "dask-cudf's groupby algorithm does not yet support "
            "`sort=True` when `split_out>1`, unless a shuffle-based "
            "algorithm is used. Please use `split_out=1`, group "
            "with `sort=False`, or set `shuffle_method=True`."
        )

    # Determine required columns to enable column projection
    required_columns = list(
        set(gb_cols).union(aggs.keys()).intersection(ddf.columns)
    )

    return aca(
        [ddf[required_columns]],
        chunk=chunk,
        chunk_kwargs=chunk_kwargs,
        combine=combine,
        combine_kwargs=combine_kwargs,
        aggregate=aggregate,
        aggregate_kwargs=aggregate_kwargs,
        token="cudf-aggregate",
        split_every=split_every,
        split_out=split_out,
        split_out_setup=split_out_on_cols,
        split_out_setup_kwargs={"cols": gb_cols},
        sort=sort,
        ignore_index=True,
    )


@_dask_cudf_performance_tracking
def _make_groupby_agg_call(
    gb, aggs, split_every, split_out, shuffle_method=None
):
    """Helper method to consolidate the common `groupby_agg` call for all
    aggregations in one place
    """

    return groupby_agg(
        gb.obj,
        gb.by,
        aggs,
        split_every=split_every,
        split_out=split_out,
        sep=gb.sep,
        sort=gb.sort,
        as_index=gb.as_index,
        shuffle_method=shuffle_method,
        **gb.dropna,
    )


@_dask_cudf_performance_tracking
def _redirect_aggs(arg):
    """Redirect aggregations to their corresponding name in cuDF"""
    redirects = {
        sum: "sum",
        max: "max",
        min: "min",
        "collect": list,
        "list": list,
    }
    if isinstance(arg, dict):
        new_arg = dict()
        for col in arg:
            if isinstance(arg[col], list):
                new_arg[col] = [redirects.get(agg, agg) for agg in arg[col]]
            elif isinstance(arg[col], dict):
                new_arg[col] = {
                    k: redirects.get(v, v) for k, v in arg[col].items()
                }
            else:
                new_arg[col] = redirects.get(arg[col], arg[col])
        return new_arg
    if isinstance(arg, list):
        return [redirects.get(agg, agg) for agg in arg]
    return redirects.get(arg, arg)


@_dask_cudf_performance_tracking
def _aggs_optimized(arg, supported: set):
    """Check that aggregations in `arg` are a subset of `supported`"""
    if isinstance(arg, (list, dict)):
        if isinstance(arg, dict):
            _global_set: set[str] = set()
            for col in arg:
                if isinstance(arg[col], list):
                    _global_set = _global_set.union(set(arg[col]))
                elif isinstance(arg[col], dict):
                    _global_set = _global_set.union(set(arg[col].values()))
                else:
                    _global_set.add(arg[col])
        else:
            _global_set = set(arg)

        return bool(_global_set.issubset(supported))
    elif isinstance(arg, (str, type)):
        return arg in supported
    return False


@_dask_cudf_performance_tracking
def _groupby_optimized(gb):
    """Check that groupby input can use dask-cudf optimized codepath"""
    return isinstance(gb.obj, DaskDataFrame) and (
        isinstance(gb.by, str)
        or (isinstance(gb.by, list) and all(isinstance(x, str) for x in gb.by))
    )


def _make_name(col_name, sep="_"):
    """Combine elements of `col_name` into a single string, or no-op if
    `col_name` is already a string
    """
    if isinstance(col_name, str):
        return col_name
    return sep.join(name for name in col_name if name != "")


@_dask_cudf_performance_tracking
def _groupby_partition_agg(df, gb_cols, aggs, columns, dropna, sort, sep):
    """Initial partition-level aggregation task.

    This is the first operation to be executed on each input
    partition in `groupby_agg`.  Depending on `aggs`, four possible
    groupby aggregations ("count", "sum", "min", and "max") are
    performed.  The result is then partitioned (by hashing `gb_cols`)
    into a number of distinct dictionary elements.  The number of
    elements in the output dictionary (`split_out`) corresponds to
    the number of partitions in the final output of `groupby_agg`.
    """

    # Modify dict for initial (partition-wise) aggregations
    _agg_dict = {}
    for col, agg_list in aggs.items():
        _agg_dict[col] = set()
        for agg in agg_list:
            if agg in ("mean", "std", "var"):
                _agg_dict[col].add("count")
                _agg_dict[col].add("sum")
            else:
                _agg_dict[col].add(agg)
        _agg_dict[col] = list(_agg_dict[col])
        if set(agg_list).intersection({"std", "var"}):
            pow2_name = _make_name((col, "pow2"), sep=sep)
            df[pow2_name] = df[col].astype("float64").pow(2)
            _agg_dict[pow2_name] = ["sum"]

    gb = df.groupby(gb_cols, dropna=dropna, as_index=False, sort=sort).agg(
        _agg_dict
    )
    output_columns = [_make_name(name, sep=sep) for name in gb.columns]
    gb.columns = output_columns
    # Return with deterministic column ordering
    return gb[sorted(output_columns)]


@_dask_cudf_performance_tracking
def _tree_node_agg(df, gb_cols, dropna, sort, sep):
    """Node in groupby-aggregation reduction tree.

    The input DataFrame (`df`) corresponds to the
    concatenated output of one or more `_groupby_partition_agg`
    tasks. In this function, "sum", "min" and/or "max" groupby
    aggregations will be used to combine the statistics for
    duplicate keys.
    """

    agg_dict = {}
    for col in df.columns:
        if col in gb_cols:
            continue
        agg = col.split(sep)[-1]
        if agg in ("count", "sum"):
            agg_dict[col] = ["sum"]
        elif agg == "list":
            agg_dict[col] = [list]
        elif agg in OPTIMIZED_AGGS:
            agg_dict[col] = [agg]
        else:
            raise ValueError(f"Unexpected aggregation: {agg}")

    gb = df.groupby(gb_cols, dropna=dropna, as_index=False, sort=sort).agg(
        agg_dict
    )

    # Don't include the last aggregation in the column names
    output_columns = [
        _make_name(name[:-1] if isinstance(name, tuple) else name, sep=sep)
        for name in gb.columns
    ]
    gb.columns = output_columns
    # Return with deterministic column ordering
    return gb[sorted(output_columns)]


@_dask_cudf_performance_tracking
def _var_agg(df, col, count_name, sum_name, pow2_sum_name, ddof=1):
    """Calculate variance (given count, sum, and sum-squared columns)."""

    # Select count, sum, and sum-squared
    n = df[count_name]
    x = df[sum_name]
    x2 = df[pow2_sum_name]

    # Use sum-squared approach to get variance
    var = x2 - x**2 / n
    div = n - ddof
    div[div < 1] = 1  # Avoid division by 0
    var /= div

    # Set appropriate NaN elements
    # (since we avoided 0-division)
    var[(n - ddof) == 0] = np.nan

    return var


@_dask_cudf_performance_tracking
def _finalize_gb_agg(
    gb_in,
    gb_cols,
    aggs,
    columns,
    final_columns,
    as_index,
    dropna,
    sort,
    sep,
    str_cols_out,
    aggs_renames,
):
    """Final aggregation task.

    This is the final operation on each output partitions
    of the `groupby_agg` algorithm.  This function must
    take care of higher-order aggregations, like "mean",
    "std" and "var".  We also need to deal with the column
    index, the row index, and final sorting behavior.
    """

    gb = _tree_node_agg(gb_in, gb_cols, dropna, sort, sep)

    # Deal with higher-order aggregations
    for col in columns:
        agg_list = aggs.get(col, [])
        agg_set = set(agg_list)
        if agg_set.intersection({"mean", "std", "var"}):
            count_name = _make_name((col, "count"), sep=sep)
            sum_name = _make_name((col, "sum"), sep=sep)
            if agg_set.intersection({"std", "var"}):
                pow2_sum_name = _make_name((col, "pow2", "sum"), sep=sep)
                var = _var_agg(gb, col, count_name, sum_name, pow2_sum_name)
                if "var" in agg_list:
                    name_var = _make_name((col, "var"), sep=sep)
                    gb[name_var] = var
                if "std" in agg_list:
                    name_std = _make_name((col, "std"), sep=sep)
                    gb[name_std] = np.sqrt(var)
                gb.drop(columns=[pow2_sum_name], inplace=True)
            if "mean" in agg_list:
                mean_name = _make_name((col, "mean"), sep=sep)
                gb[mean_name] = gb[sum_name] / gb[count_name]
            if "sum" not in agg_list:
                gb.drop(columns=[sum_name], inplace=True)
            if "count" not in agg_list:
                gb.drop(columns=[count_name], inplace=True)
        if list in agg_list:
            collect_name = _make_name((col, "list"), sep=sep)
            gb[collect_name] = gb[collect_name].list.concat()

    # Ensure sorted keys if `sort=True`
    if sort:
        gb = gb.sort_values(gb_cols)

    # Set index if necessary
    if as_index:
        gb.set_index(gb_cols, inplace=True)

    # Unflatten column names
    col_array = []
    agg_array = []
    for col in gb.columns:
        if col in gb_cols:
            col_array.append(col)
            agg_array.append("")
        else:
            name, agg = col.split(sep)
            col_array.append(name)
            agg_array.append(aggs_renames.get((name, agg), agg))
    if str_cols_out:
        gb.columns = col_array
    else:
        gb.columns = pd.MultiIndex.from_arrays([col_array, agg_array])

    return gb[final_columns]
