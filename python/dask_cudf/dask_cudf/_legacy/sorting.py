# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import warnings
from collections.abc import Iterator
from functools import wraps

import cupy
import numpy as np
import tlz as toolz

from dask import config
from dask.base import tokenize
from dask.dataframe import methods
from dask.dataframe.core import DataFrame, Index, Series
from dask.dataframe.shuffle import rearrange_by_column
from dask.highlevelgraph import HighLevelGraph
from dask.utils import M

import cudf
from cudf.api.types import _is_categorical_dtype
from cudf.utils.performance_tracking import _dask_cudf_performance_tracking

_SHUFFLE_SUPPORT = ("tasks", "p2p")  # "disk" not supported


def _deprecate_shuffle_kwarg(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        old_arg_value = kwargs.pop("shuffle", None)

        if old_arg_value is not None:
            new_arg_value = old_arg_value
            msg = (
                "the 'shuffle' keyword is deprecated, "
                "use 'shuffle_method' instead."
            )

            warnings.warn(msg, FutureWarning)
            if kwargs.get("shuffle_method") is not None:
                msg = (
                    "Can only specify 'shuffle' "
                    "or 'shuffle_method', not both."
                )
                raise TypeError(msg)
            kwargs["shuffle_method"] = new_arg_value
        return func(*args, **kwargs)

    return wrapper


@_dask_cudf_performance_tracking
def set_index_post(df, index_name, drop, column_dtype):
    df2 = df.set_index(index_name, drop=drop)
    df2.columns = df2.columns.astype(column_dtype)
    return df2


@_dask_cudf_performance_tracking
def _set_partitions_pre(s, divisions, ascending=True, na_position="last"):
    if ascending:
        partitions = divisions.searchsorted(s, side="right") - 1
    else:
        partitions = (
            len(divisions) - divisions.searchsorted(s, side="right") - 1
        )
    partitions[(partitions < 0) | (partitions >= len(divisions) - 1)] = (
        0 if ascending else (len(divisions) - 2)
    )
    partitions[s._columns[0].isnull().values] = (
        len(divisions) - 2 if na_position == "last" else 0
    )
    return partitions


@_dask_cudf_performance_tracking
def _quantile(a, q):
    n = len(a)
    if not len(a):
        return None, n
    return (
        a.quantile(q=q.tolist(), interpolation="nearest", method="table"),
        n,
    )


@_dask_cudf_performance_tracking
def merge_quantiles(finalq, qs, vals):
    """Combine several quantile calculations of different data.
    [NOTE: Same logic as dask.array merge_percentiles]
    """
    if isinstance(finalq, Iterator):
        finalq = list(finalq)
    finalq = np.array(finalq)
    qs = list(map(list, qs))
    vals = list(vals)
    vals, Ns = zip(*vals)
    Ns = list(Ns)

    L = list(zip(*[(q, val, N) for q, val, N in zip(qs, vals, Ns) if N]))
    if not L:
        raise ValueError("No non-trivial arrays found")
    qs, vals, Ns = L

    if len(vals) != len(qs) or len(Ns) != len(qs):
        raise ValueError("qs, vals, and Ns parameters must be the same length")

    # transform qs and Ns into number of observations between quantiles
    counts = []
    for q, N in zip(qs, Ns):
        count = np.empty(len(q))
        count[1:] = np.diff(q)
        count[0] = q[0]
        count *= N
        counts.append(count)

    def _append_counts(val, count):
        val["_counts"] = count
        return val

    # Sort by calculated quantile values, then number of observations.
    combined_vals_counts = cudf.core.reshape._merge_sorted(
        [*map(_append_counts, vals, counts)]
    )
    combined_counts = cupy.asnumpy(combined_vals_counts["_counts"].values)
    combined_vals = combined_vals_counts.drop(columns=["_counts"])

    # quantile-like, but scaled by total number of observations
    combined_q = np.cumsum(combined_counts)

    # rescale finalq quantiles to match combined_q
    desired_q = finalq * sum(Ns)

    # TODO: Support other interpolation methods
    # For now - Always use "nearest" for interpolation
    left = np.searchsorted(combined_q, desired_q, side="left")
    right = np.searchsorted(combined_q, desired_q, side="right") - 1
    np.minimum(left, len(combined_vals) - 1, left)  # don't exceed max index
    lower = np.minimum(left, right)
    upper = np.maximum(left, right)
    lower_residual = np.abs(combined_q[lower] - desired_q)
    upper_residual = np.abs(combined_q[upper] - desired_q)
    mask = lower_residual > upper_residual
    index = lower  # alias; we no longer need lower
    index[mask] = upper[mask]
    rv = combined_vals.iloc[index]
    return rv.reset_index(drop=True)


@_dask_cudf_performance_tracking
def _approximate_quantile(df, q):
    """Approximate quantiles of DataFrame or Series.
    [NOTE: Same logic as dask.dataframe Series quantile]
    """
    # current implementation needs q to be sorted so
    # sort if array-like, otherwise leave it alone
    q_ndarray = np.array(q)
    if q_ndarray.ndim > 0:
        q_ndarray.sort(kind="mergesort")
        q = q_ndarray

    # Lets assume we are dealing with a DataFrame throughout
    if isinstance(df, (Series, Index)):
        df = df.to_frame()
    assert isinstance(df, DataFrame)
    final_type = df._meta._constructor

    # Create metadata
    meta = df._meta_nonempty.quantile(q=q, method="table")

    # Define final action (create df with quantiles as index)
    def finalize_tsk(tsk):
        return (final_type, tsk)

    return_type = df.__class__

    # pandas/cudf uses quantile in [0, 1]
    # numpy / cupy uses [0, 100]
    qs = np.asarray(q)
    token = tokenize(df, qs)

    if len(qs) == 0:
        name = "quantiles-" + token
        empty_index = cudf.Index([], dtype=float)
        return Series(
            {
                (name, 0): final_type(
                    {col: [] for col in df.columns},
                    name=df.name,
                    index=empty_index,
                )
            },
            name,
            df._meta,
            [None, None],
        )
    else:
        new_divisions = [np.min(q), np.max(q)]

    name = "quantiles-1-" + token
    val_dsk = {
        (name, i): (_quantile, key, qs)
        for i, key in enumerate(df.__dask_keys__())
    }

    name2 = "quantiles-2-" + token
    merge_dsk = {
        (name2, 0): finalize_tsk(
            (merge_quantiles, qs, [qs] * df.npartitions, sorted(val_dsk))
        )
    }
    dsk = toolz.merge(val_dsk, merge_dsk)
    graph = HighLevelGraph.from_collections(name2, dsk, dependencies=[df])
    df = return_type(graph, name2, meta, new_divisions)

    def set_quantile_index(df):
        df.index = q
        return df

    df = df.map_partitions(set_quantile_index, meta=meta)
    return df


@_dask_cudf_performance_tracking
def quantile_divisions(df, by, npartitions):
    qn = np.linspace(0.0, 1.0, npartitions + 1).tolist()
    divisions = _approximate_quantile(df[by], qn).compute()
    columns = divisions.columns

    # TODO: Make sure divisions are correct for all dtypes..
    if (
        len(columns) == 1
        and df[columns[0]].dtype != "object"
        and not _is_categorical_dtype(df[columns[0]].dtype)
    ):
        dtype = df[columns[0]].dtype
        divisions = divisions[columns[0]].astype("int64")
        divisions.iloc[-1] += 1
        divisions = sorted(
            divisions.drop_duplicates().astype(dtype).to_arrow().tolist(),
            key=lambda x: (x is None, x),
        )
    else:
        for col in columns:
            dtype = df[col].dtype
            if dtype != "object":
                divisions[col] = divisions[col].astype("int64")
                divisions[col].iloc[-1] += 1
                divisions[col] = divisions[col].astype(dtype)
            else:
                if last := divisions[col].iloc[-1]:
                    val = chr(ord(last[0]) + 1)
                else:
                    val = "this string intentionally left empty"  # any but ""
                divisions[col].iloc[-1] = val
        divisions = divisions.drop_duplicates().sort_index()
    return divisions


@_deprecate_shuffle_kwarg
@_dask_cudf_performance_tracking
def sort_values(
    df,
    by,
    max_branch=None,
    divisions=None,
    set_divisions=False,
    ignore_index=False,
    ascending=True,
    na_position="last",
    shuffle_method=None,
    sort_function=None,
    sort_function_kwargs=None,
):
    """Sort by the given list/tuple of column names."""

    if not isinstance(ascending, bool):
        raise ValueError("ascending must be either True or False")
    if na_position not in ("first", "last"):
        raise ValueError("na_position must be either 'first' or 'last'")

    npartitions = df.npartitions
    if isinstance(by, tuple):
        by = list(by)
    elif not isinstance(by, list):
        by = [by]

    # parse custom sort function / kwargs if provided
    sort_kwargs = {
        "by": by,
        "ascending": ascending,
        "na_position": na_position,
    }
    if sort_function is None:
        sort_function = M.sort_values
    if sort_function_kwargs is not None:
        sort_kwargs.update(sort_function_kwargs)

    # handle single partition case
    if npartitions == 1:
        return df.map_partitions(sort_function, **sort_kwargs)

    # Step 1 - Calculate new divisions (if necessary)
    if divisions is None:
        divisions = quantile_divisions(df, by, npartitions)

    # Step 2 - Perform repartitioning shuffle
    meta = df._meta._constructor_sliced([0])
    if not isinstance(divisions, (cudf.Series, cudf.DataFrame)):
        dtype = df[by[0]].dtype
        divisions = df._meta._constructor_sliced(divisions, dtype=dtype)

    partitions = df[by].map_partitions(
        _set_partitions_pre,
        divisions=divisions,
        ascending=ascending,
        na_position=na_position,
        meta=meta,
    )

    df2 = df.assign(_partitions=partitions)
    df3 = rearrange_by_column(
        df2,
        "_partitions",
        max_branch=max_branch,
        npartitions=len(divisions) - 1,
        shuffle_method=_get_shuffle_method(shuffle_method),
        ignore_index=ignore_index,
    ).drop(columns=["_partitions"])
    df3.divisions = (None,) * (df3.npartitions + 1)

    # Step 3 - Return final sorted df
    df4 = df3.map_partitions(sort_function, **sort_kwargs)
    if not isinstance(divisions, cudf.DataFrame) and set_divisions:
        # Can't have multi-column divisions elsewhere in dask (yet)
        df4.divisions = tuple(methods.tolist(divisions))

    return df4


def get_default_shuffle_method():
    # Note that `dask.utils.get_default_shuffle_method`
    # will return "p2p" by default when a distributed
    # client is present. Dask-cudf supports "p2p", but
    # will not use it by default (yet)
    default = config.get("dataframe.shuffle.method", "tasks")
    if default not in _SHUFFLE_SUPPORT:
        default = "tasks"
    return default


def _get_shuffle_method(shuffle_method):
    # Utility to set the shuffle_method-kwarg default
    # and to validate user-specified options
    shuffle_method = shuffle_method or get_default_shuffle_method()
    if shuffle_method not in _SHUFFLE_SUPPORT:
        raise ValueError(
            "Dask-cudf only supports the following shuffle "
            f"methods: {_SHUFFLE_SUPPORT}. Got shuffle_method={shuffle_method}"
        )

    return shuffle_method
