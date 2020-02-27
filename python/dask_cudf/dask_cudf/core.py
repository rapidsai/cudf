# Copyright (c) 2018, NVIDIA CORPORATION.
import warnings
from operator import getitem

import numpy as np
import pandas as pd
from toolz import partition_all

import dask
import dask.dataframe as dd
from dask import compute
from dask.base import normalize_token, tokenize
from dask.compatibility import apply
from dask.context import _globals
from dask.core import flatten
from dask.dataframe import from_delayed
from dask.dataframe.core import Scalar, handle_out, map_partitions
from dask.dataframe.utils import is_series_like, raise_on_meta_error
from dask.delayed import delayed
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import cull, fuse
from dask.utils import (
    M,
    OperatorMethodMixin,
    derived_from,
    funcname,
    parse_bytes,
)

import cudf
import cudf._lib as libcudf

from dask_cudf import batcher_sortnet
from dask_cudf.accessor import (
    CachedAccessor,
    CategoricalAccessor,
    DatetimeAccessor,
)


def optimize(dsk, keys, **kwargs):
    flatkeys = list(flatten(keys)) if isinstance(keys, list) else [keys]
    dsk, dependencies = cull(dsk, flatkeys)
    dsk, dependencies = fuse(
        dsk,
        keys,
        dependencies=dependencies,
        ave_width=_globals.get("fuse_ave_width", 1),
    )
    dsk, _ = cull(dsk, keys)
    return dsk


def finalize(results):
    if results and isinstance(
        results[0], (cudf.DataFrame, cudf.Series, cudf.Index, cudf.MultiIndex)
    ):
        return cudf.concat(results)
    return results


class _Frame(dd.core._Frame, OperatorMethodMixin):
    """ Superclass for DataFrame and Series

    Parameters
    ----------
    dsk : dict
        The dask graph to compute this DataFrame
    name : str
        The key prefix that specifies which keys in the dask comprise this
        particular DataFrame / Series
    meta : cudf.DataFrame, cudf.Series, or cudf.Index
        An empty cudf object with names, dtypes, and indices matching the
        expected output.
    divisions : tuple of index values
        Values along which we partition our blocks on the index
    """

    __dask_scheduler__ = staticmethod(dask.get)
    __dask_optimize__ = staticmethod(optimize)

    def __dask_postcompute__(self):
        return finalize, ()

    def __dask_postpersist__(self):
        return type(self), (self._name, self._meta, self.divisions)

    def __init__(self, dsk, name, meta, divisions):
        if not isinstance(dsk, HighLevelGraph):
            dsk = HighLevelGraph.from_collections(name, dsk, dependencies=[])
        self.dask = dsk
        self._name = name
        meta = dd.core.make_meta(meta)
        if not isinstance(meta, self._partition_type):
            raise TypeError(
                "Expected meta to specify type {0}, got type "
                "{1}".format(
                    self._partition_type.__name__, type(meta).__name__
                )
            )
        self._meta = meta
        self.divisions = tuple(divisions)

    def __getstate__(self):
        return (self.dask, self._name, self._meta, self.divisions)

    def __setstate__(self, state):
        self.dask, self._name, self._meta, self.divisions = state

    def __repr__(self):
        s = "<dask_cudf.%s | %d tasks | %d npartitions>"
        return s % (type(self).__name__, len(self.dask), self.npartitions)

    def to_dask_dataframe(self):
        """Create a dask.dataframe object from a dask_cudf object"""
        return self.map_partitions(M.to_pandas)


concat = dd.concat


normalize_token.register(_Frame, lambda a: a._name)


class DataFrame(_Frame, dd.core.DataFrame):
    _partition_type = cudf.DataFrame

    def _assign_column(self, k, v):
        def assigner(df, k, v):
            out = df.copy()
            out[k] = v
            return out

        meta = assigner(self._meta, k, dd.core.make_meta(v))
        return self.map_partitions(assigner, k, v, meta=meta)

    def apply_rows(self, func, incols, outcols, kwargs={}, cache_key=None):
        import uuid

        if cache_key is None:
            cache_key = uuid.uuid4()

        def do_apply_rows(df, func, incols, outcols, kwargs):
            return df.apply_rows(
                func, incols, outcols, kwargs, cache_key=cache_key
            )

        meta = do_apply_rows(self._meta, func, incols, outcols, kwargs)
        return self.map_partitions(
            do_apply_rows, func, incols, outcols, kwargs, meta=meta
        )

    def merge(self, other, **kwargs):
        if kwargs.pop("shuffle", "tasks") != "tasks":
            raise ValueError(
                "Dask-cudf only supports task based shuffling, got %s"
                % kwargs["shuffle"]
            )
        on = kwargs.pop("on", None)
        if isinstance(on, tuple):
            on = list(on)
        return super().merge(other, on=on, shuffle="tasks", **kwargs)

    def join(self, other, **kwargs):
        if kwargs.pop("shuffle", "tasks") != "tasks":
            raise ValueError(
                "Dask-cudf only supports task based shuffling, got %s"
                % kwargs["shuffle"]
            )

        # CuDF doesn't support "right" join yet
        how = kwargs.pop("how", "left")
        if how == "right":
            return other.join(other=self, how="left", **kwargs)

        on = kwargs.pop("on", None)
        if isinstance(on, tuple):
            on = list(on)
        return super().join(other, how=how, on=on, shuffle="tasks", **kwargs)

    def set_index(self, other, **kwargs):
        if kwargs.pop("shuffle", "tasks") != "tasks":
            raise ValueError(
                "Dask-cudf only supports task based shuffling, got %s"
                % kwargs["shuffle"]
            )
        return super().set_index(other, shuffle="tasks", **kwargs)

    def sort_values(self, by, ignore_index=False):
        """Sort by the given column

        Parameter
        ---------
        by : str
        """
        if self.npartitions == 1:
            df = self.map_partitions(M.sort_values, by)
        else:
            parts = self.to_delayed()
            sorted_parts = batcher_sortnet.sort_delayed_frame(parts, by)
            df = from_delayed(sorted_parts, meta=self._meta)
        if ignore_index:
            return df.reset_index(drop=True)
        return df

    def sort_values_binned(self, by):
        """Sorty by the given column and ensure that the same key
        doesn't spread across multiple partitions.
        """
        # Get sorted partitions
        parts = self.sort_values(by=by).to_delayed()

        # Get unique keys in each partition
        @delayed
        def get_unique(p):
            return set(p[by].unique())

        uniques = list(compute(*map(get_unique, parts)))

        joiner = {}
        for i in range(len(uniques)):
            joiner[i] = to_join = {}
            for j in range(i + 1, len(uniques)):
                intersect = uniques[i] & uniques[j]
                # If the keys intersect
                if intersect:
                    # Remove keys
                    uniques[j] -= intersect
                    to_join[j] = frozenset(intersect)
                else:
                    break

        @delayed
        def join(df, other, keys):
            others = [
                other.query("{by}==@k".format(by=by)) for k in sorted(keys)
            ]
            return cudf.concat([df] + others)

        @delayed
        def drop(df, keep_keys):
            locvars = locals()
            for i, k in enumerate(keep_keys):
                locvars["k{}".format(i)] = k

            conds = [
                "{by}==@k{i}".format(by=by, i=i) for i in range(len(keep_keys))
            ]
            expr = " or ".join(conds)
            return df.query(expr)

        for i in range(len(parts)):
            if uniques[i]:
                parts[i] = drop(parts[i], uniques[i])
                for joinee, intersect in joiner[i].items():
                    parts[i] = join(parts[i], parts[joinee], intersect)

        results = [p for i, p in enumerate(parts) if uniques[i]]
        return from_delayed(results, meta=self._meta).reset_index()

    def to_parquet(self, path, *args, **kwargs):
        """ Calls dask.dataframe.io.to_parquet with CudfEngine backend """
        from dask_cudf.io import to_parquet

        return to_parquet(self, path, *args, **kwargs)

    def to_orc(self, path, **kwargs):
        """ Calls dask_cudf.io.to_orc """
        from dask_cudf.io import to_orc

        return to_orc(self, path, **kwargs)

    @derived_from(pd.DataFrame)
    def var(
        self,
        axis=None,
        skipna=True,
        ddof=1,
        split_every=False,
        dtype=None,
        out=None,
    ):
        axis = self._validate_axis(axis)
        meta = self._meta_nonempty.var(axis=axis, skipna=skipna)
        if axis == 1:
            result = map_partitions(
                M.var,
                self,
                meta=meta,
                token=self._token_prefix + "var",
                axis=axis,
                skipna=skipna,
                ddof=ddof,
            )
            return handle_out(out, result)

        else:
            num = self._get_numeric_data()
            x = 1.0 * num.sum(skipna=skipna, split_every=split_every)
            x2 = 1.0 * (num ** 2).sum(skipna=skipna, split_every=split_every)
            n = num.count(split_every=split_every)
            name = self._token_prefix + "var"
            result = map_partitions(
                var_aggregate, x2, x, n, token=name, meta=meta, ddof=ddof
            )
            if isinstance(self, DataFrame):
                result.divisions = (min(self.columns), max(self.columns))
            return handle_out(out, result)


def sum_of_squares(x):
    x = x.astype("f8")._column
    outcol = libcudf.reduce.reduce("sum_of_squares", x)
    return cudf.Series(outcol)


def var_aggregate(x2, x, n, ddof):
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = (x2 / n) - (x / n) ** 2
        if ddof != 0:
            result = result * n / (n - ddof)
        return result
    except ZeroDivisionError:
        return np.float64(np.nan)


def nlargest_agg(x, **kwargs):
    return cudf.concat(x).nlargest(**kwargs)


def nsmallest_agg(x, **kwargs):
    return cudf.concat(x).nsmallest(**kwargs)


def unique_k_agg(x, **kwargs):
    return cudf.concat(x).unique_k(**kwargs)


class Series(_Frame, dd.core.Series):
    _partition_type = cudf.Series

    def count(self, split_every=False):
        return reduction(
            self,
            chunk=M.count,
            aggregate=np.sum,
            split_every=split_every,
            meta="i8",
        )

    def mean(self, split_every=False):
        sum = self.sum(split_every=split_every)
        n = self.count(split_every=split_every)
        return sum / n

    def unique_k(self, k, split_every=None):
        return reduction(
            self,
            chunk=M.unique_k,
            aggregate=unique_k_agg,
            meta=self._meta,
            token="unique-k",
            split_every=split_every,
            k=k,
        )

    @derived_from(pd.DataFrame)
    def var(
        self,
        axis=None,
        skipna=True,
        ddof=1,
        split_every=False,
        dtype=None,
        out=None,
    ):
        axis = self._validate_axis(axis)
        meta = self._meta_nonempty.var(axis=axis, skipna=skipna)
        if axis == 1:
            result = map_partitions(
                M.var,
                self,
                meta=meta,
                token=self._token_prefix + "var",
                axis=axis,
                skipna=skipna,
                ddof=ddof,
            )
            return handle_out(out, result)

        else:
            num = self._get_numeric_data()
            x = 1.0 * num.sum(skipna=skipna, split_every=split_every)
            x2 = 1.0 * (num ** 2).sum(skipna=skipna, split_every=split_every)
            n = num.count(split_every=split_every)
            name = self._token_prefix + "var"
            result = map_partitions(
                var_aggregate, x2, x, n, token=name, meta=meta, ddof=ddof
            )
            if isinstance(self, DataFrame):
                result.divisions = (min(self.columns), max(self.columns))
            return handle_out(out, result)

    # ----------------------------------------------------------------------
    # Accessor Methods
    # ----------------------------------------------------------------------
    dt = CachedAccessor("dt", DatetimeAccessor)
    cat = CachedAccessor("cat", CategoricalAccessor)


class Index(Series, dd.core.Index):
    _partition_type = cudf.Index


def splits_divisions_sorted_cudf(df, chunksize):
    segments = list(df.index.find_segments().to_array())
    segments.append(len(df) - 1)

    splits = [0]
    last = current_size = 0
    for s in segments:
        size = s - last
        last = s
        current_size += size
        if current_size >= chunksize:
            splits.append(s)
            current_size = 0
    # Ensure end is included
    if splits[-1] != segments[-1]:
        splits.append(segments[-1])
    divisions = tuple(df.index.take(np.array(splits)).values)
    splits[-1] += 1  # Offset to extract to end

    return splits, divisions


def _extract_meta(x):
    """
    Extract internal cache data (``_meta``) from dask_cudf objects
    """
    if isinstance(x, (Scalar, _Frame)):
        return x._meta
    elif isinstance(x, list):
        return [_extract_meta(_x) for _x in x]
    elif isinstance(x, tuple):
        return tuple([_extract_meta(_x) for _x in x])
    elif isinstance(x, dict):
        return {k: _extract_meta(v) for k, v in x.items()}
    return x


def _emulate(func, *args, **kwargs):
    """
    Apply a function using args / kwargs. If arguments contain dd.DataFrame /
    dd.Series, using internal cache (``_meta``) for calculation
    """
    with raise_on_meta_error(funcname(func)):
        return func(*_extract_meta(args), **_extract_meta(kwargs))


def align_partitions(args):
    """Align partitions between dask_cudf objects.

    Note that if all divisions are unknown, but have equal npartitions, then
    they will be passed through unchanged."""
    dfs = [df for df in args if isinstance(df, _Frame)]
    if not dfs:
        return args

    divisions = dfs[0].divisions
    if not all(df.divisions == divisions for df in dfs):
        raise NotImplementedError("Aligning mismatched partitions")
    return args


def reduction(
    args,
    chunk=None,
    aggregate=None,
    combine=None,
    meta=None,
    token=None,
    chunk_kwargs=None,
    aggregate_kwargs=None,
    combine_kwargs=None,
    split_every=None,
    **kwargs,
):
    """Generic tree reduction operation.

    Parameters
    ----------
    args :
        Positional arguments for the `chunk` function. All `dask.dataframe`
        objects should be partitioned and indexed equivalently.
    chunk : function [block-per-arg] -> block
        Function to operate on each block of data
    aggregate : function list-of-blocks -> block
        Function to operate on the list of results of chunk
    combine : function list-of-blocks -> block, optional
        Function to operate on intermediate lists of results of chunk
        in a tree-reduction. If not provided, defaults to aggregate.
    $META
    token : str, optional
        The name to use for the output keys.
    chunk_kwargs : dict, optional
        Keywords for the chunk function only.
    aggregate_kwargs : dict, optional
        Keywords for the aggregate function only.
    combine_kwargs : dict, optional
        Keywords for the combine function only.
    split_every : int, optional
        Group partitions into groups of this size while performing a
        tree-reduction. If set to False, no tree-reduction will be used,
        and all intermediates will be concatenated and passed to ``aggregate``.
        Default is 8.
    kwargs :
        All remaining keywords will be passed to ``chunk``, ``aggregate``, and
        ``combine``.
    """
    if chunk_kwargs is None:
        chunk_kwargs = dict()
    if aggregate_kwargs is None:
        aggregate_kwargs = dict()
    chunk_kwargs.update(kwargs)
    aggregate_kwargs.update(kwargs)

    if combine is None:
        if combine_kwargs:
            raise ValueError("`combine_kwargs` provided with no `combine`")
        combine = aggregate
        combine_kwargs = aggregate_kwargs
    else:
        if combine_kwargs is None:
            combine_kwargs = dict()
        combine_kwargs.update(kwargs)

    if not isinstance(args, (tuple, list)):
        args = [args]

    npartitions = set(
        arg.npartitions for arg in args if isinstance(arg, _Frame)
    )
    if len(npartitions) > 1:
        raise ValueError("All arguments must have same number of partitions")
    npartitions = npartitions.pop()

    if split_every is None:
        split_every = 8
    elif split_every is False:
        split_every = npartitions
    elif split_every < 2 or not isinstance(split_every, int):
        raise ValueError("split_every must be an integer >= 2")

    token_key = tokenize(
        token or (chunk, aggregate),
        meta,
        args,
        chunk_kwargs,
        aggregate_kwargs,
        combine_kwargs,
        split_every,
    )

    # Chunk
    a = "{0}-chunk-{1}".format(token or funcname(chunk), token_key)
    if len(args) == 1 and isinstance(args[0], _Frame) and not chunk_kwargs:
        dsk = {
            (a, 0, i): (chunk, key)
            for i, key in enumerate(args[0].__dask_keys__())
        }
    else:
        dsk = {
            (a, 0, i): (
                apply,
                chunk,
                [(x._name, i) if isinstance(x, _Frame) else x for x in args],
                chunk_kwargs,
            )
            for i in range(args[0].npartitions)
        }

    # Combine
    b = "{0}-combine-{1}".format(token or funcname(combine), token_key)
    k = npartitions
    depth = 0
    while k > split_every:
        for part_i, inds in enumerate(partition_all(split_every, range(k))):
            conc = (list, [(a, depth, i) for i in inds])
            dsk[(b, depth + 1, part_i)] = (
                (apply, combine, [conc], combine_kwargs)
                if combine_kwargs
                else (combine, conc)
            )
        k = part_i + 1
        a = b
        depth += 1

    # Aggregate
    b = "{0}-agg-{1}".format(token or funcname(aggregate), token_key)
    conc = (list, [(a, depth, i) for i in range(k)])
    if aggregate_kwargs:
        dsk[(b, 0)] = (apply, aggregate, [conc], aggregate_kwargs)
    else:
        dsk[(b, 0)] = (aggregate, conc)

    if meta is None:
        meta_chunk = _emulate(apply, chunk, args, chunk_kwargs)
        meta = _emulate(apply, aggregate, [[meta_chunk]], aggregate_kwargs)
    meta = dd.core.make_meta(meta)

    for arg in args:
        if isinstance(arg, _Frame):
            dsk.update(arg.dask)

    return dd.core.new_dd_object(dsk, b, meta, (None, None))


def _total_mem_usage(df):
    mem_usage = df.memory_usage(deep=True)
    if is_series_like(mem_usage):
        mem_usage = mem_usage.sum()
    return mem_usage


def _split_func(df, k):
    """ Split dataframe along index boundaries.
        If boundaries don't exist, use linear division.
    """
    df = df.sort_index()
    q = [float(i + 1) / k for i in range(k - 1)]
    splits = (
        cudf.Series(df.index)
        .quantile(q, interpolation="nearest")
        .astype(df.index.dtype)
    )
    divisions = df.index.searchsorted(splits, side="left").tolist()
    divisions.insert(0, 0)
    divisions.append(len(df))
    if len(set(divisions)) != len(divisions):
        i = 0
        while i < len(divisions) - 2:
            j = i
            while divisions[i] == divisions[j + 1]:
                j += 1
            if j != i:
                for ind, val in enumerate(
                    np.linspace(
                        divisions[i], divisions[j + 1], j - i + 2
                    ).astype(int)[:-1]
                ):
                    divisions[i + ind] = val
                i = j
            else:
                i += 1
    return {i: df.iloc[divisions[i] : divisions[i + 1]] for i in range(k)}


def split_partitions(df, max_size=None, max_length=None):
    """ Split each partition of a dask DataFrame (by rule)

    Parameters
    ----------
    df: DataFrame or Series
    max_size: int or str
        Maximum data size desired in an output partition
    max_length: int or str
        Maximum row-count desired in an output partition

    Note: User must specify max_size or max_length.
    If both are specified, max_length is ignored.
    """

    # Ensure split criteria is specified
    if not (bool(max_size) or bool(max_length)):
        raise ValueError("Must specify max_size or max_length")

    # Determine the number of splits in each partition
    if max_size is not None:

        if isinstance(max_size, str):
            max_size = parse_bytes(max_size)
        max_size = int(max_size)
        mem_usages = df.map_partitions(_total_mem_usage).compute()
        nsplits = 1 + mem_usages // max_size
    else:
        max_length = int(max_length)
        lengths = df.map_partitions(lambda x: len(x)).compute()
        nsplits = 1 + lengths // max_length

    # Return if none of the partitions need splitting
    if not np.any(nsplits > 1):
        return df

    # Build new Dask DataFrame
    dsk = {}
    new_name = "repartition-split-{}-{}".format(max_length, tokenize(df))
    split_name = "split-{}".format(tokenize(df, nsplits))
    j = 0
    for i, k in enumerate(nsplits):
        if k == 1:
            dsk[new_name, j] = (df._name, i)
            j += 1
        else:
            dsk[split_name, i] = (_split_func, (df._name, i), k)
            for jj in range(k):
                dsk[new_name, j] = (getitem, (split_name, i), jj)
                j += 1

    divisions = [None] * (1 + sum(nsplits))
    graph = HighLevelGraph.from_collections(new_name, dsk, dependencies=[df])
    return dd.core.new_dd_object(graph, new_name, df._meta, divisions)


from_cudf = dd.from_pandas


def from_dask_dataframe(df):
    return df.map_partitions(cudf.from_pandas)


for name in [
    "add",
    "sub",
    "mul",
    "truediv",
    "floordiv",
    "mod",
    "pow",
    "radd",
    "rsub",
    "rmul",
    "rtruediv",
    "rfloordiv",
    "rmod",
    "rpow",
]:
    meth = getattr(cudf.DataFrame, name)
    DataFrame._bind_operator_method(name, meth)

    meth = getattr(cudf.Series, name)
    Series._bind_operator_method(name, meth)

for name in ["lt", "gt", "le", "ge", "ne", "eq"]:

    meth = getattr(cudf.Series, name)
    Series._bind_comparison_method(name, meth)
