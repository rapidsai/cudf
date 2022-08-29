# Copyright (c) 2020-2022, NVIDIA CORPORATION.

import math
from operator import getitem
from typing import Set

import numpy as np
import pandas as pd

from dask.base import tokenize
from dask.dataframe.core import (
    DataFrame as DaskDataFrame,
    _concat,
    hash_shard,
    new_dd_object,
    split_out_on_cols,
)
from dask.dataframe.groupby import DataFrameGroupBy, SeriesGroupBy
from dask.highlevelgraph import HighLevelGraph

import cudf
from cudf.utils.utils import _dask_cudf_nvtx_annotate

CUMULATIVE_AGGS = (
    "cumsum",
    "cumcount",
)

AGGS = (
    "count",
    "mean",
    "std",
    "var",
    "sum",
    "min",
    "max",
    "collect",
    "first",
    "last",
)

SUPPORTED_AGGS = (*AGGS, *CUMULATIVE_AGGS)


def _check_groupby_supported(func):
    """
    Decorator for dask-cudf's groupby methods that returns the dask-cudf
    method if the groupby object is supported, otherwise reverting to the
    upstream Dask method
    """

    def wrapper(*args, **kwargs):
        gb = args[0]
        if _groupby_supported(gb):
            return func(*args, **kwargs)
        # note that we use upstream Dask's default kwargs for this call if
        # none are specified; this shouldn't be an issue as those defaults are
        # consistent with dask-cudf
        return getattr(super(type(gb), gb), func.__name__)(*args[1:], **kwargs)

    return wrapper


class CudfDataFrameGroupBy(DataFrameGroupBy):
    @_dask_cudf_nvtx_annotate
    def __init__(self, *args, **kwargs):
        self.sep = kwargs.pop("sep", "___")
        self.as_index = kwargs.pop("as_index", True)
        super().__init__(*args, **kwargs)

    @_dask_cudf_nvtx_annotate
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

    @_dask_cudf_nvtx_annotate
    def _make_groupby_method_aggs(self, agg_name):
        """Create aggs dictionary for aggregation methods"""

        if isinstance(self.by, list):
            return {c: agg_name for c in self.obj.columns if c not in self.by}
        return {c: agg_name for c in self.obj.columns if c != self.by}

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def count(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            self._make_groupby_method_aggs("count"),
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def mean(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            self._make_groupby_method_aggs("mean"),
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def std(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            self._make_groupby_method_aggs("std"),
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def var(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            self._make_groupby_method_aggs("var"),
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def sum(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            self._make_groupby_method_aggs("sum"),
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def min(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            self._make_groupby_method_aggs("min"),
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def max(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            self._make_groupby_method_aggs("max"),
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def collect(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            self._make_groupby_method_aggs("collect"),
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def first(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            self._make_groupby_method_aggs("first"),
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def last(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            self._make_groupby_method_aggs("last"),
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )

    @_dask_cudf_nvtx_annotate
    def aggregate(self, arg, split_every=None, split_out=1, shuffle=None):
        if arg == "size":
            return self.size()

        arg = _redirect_aggs(arg)

        if _groupby_supported(self) and _aggs_supported(arg, SUPPORTED_AGGS):
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
                **self.dropna,
            )

        return super().aggregate(
            arg,
            split_every=split_every,
            split_out=split_out,
            # TODO: Change following line to `shuffle=shuffle,`
            # when dask_cudf is pinned to dask>2022.8.0
            **({} if shuffle is None else {"shuffle": shuffle}),
        )


class CudfSeriesGroupBy(SeriesGroupBy):
    @_dask_cudf_nvtx_annotate
    def __init__(self, *args, **kwargs):
        self.sep = kwargs.pop("sep", "___")
        self.as_index = kwargs.pop("as_index", True)
        super().__init__(*args, **kwargs)

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def count(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            {self._slice: "count"},
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )[self._slice]

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def mean(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            {self._slice: "mean"},
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )[self._slice]

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def std(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            {self._slice: "std"},
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )[self._slice]

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def var(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            {self._slice: "var"},
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )[self._slice]

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def sum(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            {self._slice: "sum"},
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )[self._slice]

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def min(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            {self._slice: "min"},
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )[self._slice]

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def max(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            {self._slice: "max"},
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )[self._slice]

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def collect(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            {self._slice: "collect"},
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )[self._slice]

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def first(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            {self._slice: "first"},
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )[self._slice]

    @_dask_cudf_nvtx_annotate
    @_check_groupby_supported
    def last(self, split_every=None, split_out=1):
        return groupby_agg(
            self.obj,
            self.by,
            {self._slice: "last"},
            split_every=split_every,
            split_out=split_out,
            sep=self.sep,
            sort=self.sort,
            as_index=self.as_index,
            **self.dropna,
        )[self._slice]

    @_dask_cudf_nvtx_annotate
    def aggregate(self, arg, split_every=None, split_out=1, shuffle=None):
        if arg == "size":
            return self.size()

        arg = _redirect_aggs(arg)

        if not isinstance(arg, dict):
            arg = {self._slice: arg}

        if _groupby_supported(self) and _aggs_supported(arg, SUPPORTED_AGGS):
            return groupby_agg(
                self.obj,
                self.by,
                arg,
                split_every=split_every,
                split_out=split_out,
                sep=self.sep,
                sort=self.sort,
                as_index=self.as_index,
                **self.dropna,
            )[self._slice]

        return super().aggregate(
            arg,
            split_every=split_every,
            split_out=split_out,
            # TODO: Change following line to `shuffle=shuffle,`
            # when dask_cudf is pinned to dask>2022.8.0
            **({} if shuffle is None else {"shuffle": shuffle}),
        )


@_dask_cudf_nvtx_annotate
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
):
    """Optimized groupby aggregation for Dask-CuDF.

    This aggregation algorithm only supports the following options:

    - "count"
    - "mean"
    - "std"
    - "var"
    - "sum"
    - "min"
    - "max"
    - "collect"
    - "first"
    - "last"

    This "optimized" approach is more performant than the algorithm
    in `dask.dataframe`, because it allows the cudf backend to
    perform multiple aggregations at once.
    """
    # Assert that aggregations are supported
    aggs = _redirect_aggs(aggs_in)
    if not _aggs_supported(aggs, SUPPORTED_AGGS):
        raise ValueError(
            f"Supported aggs include {SUPPORTED_AGGS} for groupby_agg API. "
            f"Aggregations must be specified with dict or list syntax."
        )

    # Deal with default split_out and split_every params
    if split_every is False:
        split_every = ddf.npartitions
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

    # Begin graph construction
    dsk = {}
    token = tokenize(ddf, gb_cols, aggs)
    partition_agg_name = "groupby_partition_agg-" + token
    tree_reduce_name = "groupby_tree_reduce-" + token
    gb_agg_name = "groupby_agg-" + token
    for p in range(ddf.npartitions):
        # Perform groupby aggregation on each partition.
        # Split each result into `split_out` chunks (by hashing `gb_cols`)
        dsk[(partition_agg_name, p)] = (
            _groupby_partition_agg,
            (ddf._name, p),
            gb_cols,
            aggs,
            columns,
            split_out,
            dropna,
            sort,
            sep,
        )
        # Pick out each chunk using `getitem`
        for s in range(split_out):
            dsk[(tree_reduce_name, p, s, 0)] = (
                getitem,
                (partition_agg_name, p),
                s,
            )

    # Build reduction tree
    parts = ddf.npartitions
    widths = [parts]
    while parts > 1:
        parts = math.ceil(parts / split_every)
        widths.append(parts)
    height = len(widths)
    for s in range(split_out):
        for depth in range(1, height):
            for group in range(widths[depth]):

                p_max = widths[depth - 1]
                lstart = split_every * group
                lstop = min(lstart + split_every, p_max)
                node_list = [
                    (tree_reduce_name, p, s, depth - 1)
                    for p in range(lstart, lstop)
                ]

                dsk[(tree_reduce_name, group, s, depth)] = (
                    _tree_node_agg,
                    node_list,
                    gb_cols,
                    split_out,
                    dropna,
                    sort,
                    sep,
                )

    # Final output partitions.
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
    for s in range(split_out):
        dsk[(gb_agg_name, s)] = (
            _finalize_gb_agg,
            (tree_reduce_name, 0, s, height - 1),
            gb_cols,
            aggs,
            columns,
            _meta.columns,
            as_index,
            sort,
            sep,
            str_cols_out,
            aggs_renames,
        )

    divisions = [None] * (split_out + 1)
    graph = HighLevelGraph.from_collections(
        gb_agg_name, dsk, dependencies=[ddf]
    )
    return new_dd_object(graph, gb_agg_name, _meta, divisions)


@_dask_cudf_nvtx_annotate
def _redirect_aggs(arg):
    """Redirect aggregations to their corresponding name in cuDF"""
    redirects = {
        sum: "sum",
        max: "max",
        min: "min",
        list: "collect",
        "list": "collect",
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


@_dask_cudf_nvtx_annotate
def _aggs_supported(arg, supported: set):
    """Check that aggregations in `arg` are a subset of `supported`"""
    if isinstance(arg, (list, dict)):
        if isinstance(arg, dict):
            _global_set: Set[str] = set()
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
    elif isinstance(arg, str):
        return arg in supported
    return False


@_dask_cudf_nvtx_annotate
def _groupby_supported(gb):
    """Check that groupby input is supported by dask-cudf"""
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


@_dask_cudf_nvtx_annotate
def _groupby_partition_agg(
    df, gb_cols, aggs, columns, split_out, dropna, sort, sep
):
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
    gb.columns = [_make_name(name, sep=sep) for name in gb.columns]

    if split_out == 1:
        output = {0: gb.copy(deep=False)}
    elif hasattr(gb, "partition_by_hash"):
        # For cudf, we can use `partition_by_hash` method
        output = {}
        for j, split in enumerate(
            gb.partition_by_hash(gb_cols, split_out, keep_index=False)
        ):
            output[j] = split
        del gb
    else:
        # Dask-Dataframe (Pandas) support
        output = hash_shard(
            gb,
            split_out,
            split_out_setup=split_out_on_cols,
            split_out_setup_kwargs={"cols": gb_cols},
        )
        del gb
    return output


@_dask_cudf_nvtx_annotate
def _tree_node_agg(dfs, gb_cols, split_out, dropna, sort, sep):
    """Node in groupby-aggregation reduction tree.

    Following the initial `_groupby_partition_agg` tasks,
    the `groupby_agg` algorithm will perform a tree reduction
    to combine the data from the input partitions into
    `split_out` different output partitions.  For each node in
    the reduction tree, the input DataFrame objects are
    concatenated, and "sum", "min" and/or "max" groupby
    aggregations are used to combine the necessary statistics.
    """

    df = _concat(dfs, ignore_index=True)
    agg_dict = {}
    for col in df.columns:
        if col in gb_cols:
            continue
        agg = col.split(sep)[-1]
        if agg in ("count", "sum"):
            agg_dict[col] = ["sum"]
        elif agg in SUPPORTED_AGGS:
            agg_dict[col] = [agg]
        else:
            raise ValueError(f"Unexpected aggregation: {agg}")

    gb = df.groupby(gb_cols, dropna=dropna, as_index=False, sort=sort).agg(
        agg_dict
    )

    # Don't include the last aggregation in the column names
    gb.columns = [
        _make_name(name[:-1] if isinstance(name, tuple) else name, sep=sep)
        for name in gb.columns
    ]
    return gb


@_dask_cudf_nvtx_annotate
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


@_dask_cudf_nvtx_annotate
def _finalize_gb_agg(
    gb,
    gb_cols,
    aggs,
    columns,
    final_columns,
    as_index,
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
        if "collect" in agg_list:
            collect_name = _make_name((col, "collect"), sep=sep)
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
