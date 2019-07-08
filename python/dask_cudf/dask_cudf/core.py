# Copyright (c) 2018, NVIDIA CORPORATION.
import warnings
from collections import OrderedDict

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
from dask.dataframe.utils import raise_on_meta_error
from dask.delayed import delayed
from dask.optimization import cull, fuse
from dask.utils import M, OperatorMethodMixin, derived_from, funcname

import cudf
import cudf.bindings.reduce as cpp_reduce

from dask_cudf import batcher_sortnet, join_impl
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
    return cudf.concat(results)


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
        self._meta = dd.core.make_meta(meta)
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

    def merge(
        self,
        other,
        on=None,
        how="left",
        left_index=False,
        right_index=False,
        suffixes=("_x", "_y"),
    ):
        """Merging two dataframes on the column(s) indicated in *on*.
        """
        if (
            left_index
            or right_index
            or not dask.is_dask_collection(other)
            or self.npartitions == 1
            and how in ("inner", "right")
            or other.npartitions == 1
            and how in ("inner", "left")
        ):
            return dd.merge(
                self,
                other,
                how=how,
                suffixes=suffixes,
                left_index=left_index,
                right_index=right_index,
            )

        if not on and not left_index and not right_index:
            on = [c for c in self.columns if c in other.columns]
            if not on:
                left_index = right_index = True

        return join_impl.join_frames(
            left=self,
            right=other,
            on=on,
            how=how,
            lsuffix=suffixes[0],
            rsuffix=suffixes[1],
        )

    def join(self, other, how="left", lsuffix="", rsuffix=""):
        """Join two datatframes

        *on* is not supported.
        """
        if how == "right":
            return other.join(
                other=self, how="left", lsuffix=rsuffix, rsuffix=lsuffix
            )

        same_names = set(self.columns) & set(other.columns)
        if same_names and not (lsuffix or rsuffix):
            raise ValueError(
                "there are overlapping columns but "
                "lsuffix and rsuffix are not defined"
            )

        left, leftuniques = self._align_divisions()
        right, rightuniques = other._align_to_indices(leftuniques)

        leftparts = left.to_delayed()
        rightparts = right.to_delayed()

        @delayed
        def part_join(left, right, how):
            return left.join(
                right, how=how, sort=True, lsuffix=lsuffix, rsuffix=rsuffix
            )

        def inner_selector():
            pivot = 0
            for i in range(len(leftparts)):
                for j in range(pivot, len(rightparts)):
                    if leftuniques[i] & rightuniques[j]:
                        yield leftparts[i], rightparts[j]
                        pivot = j + 1
                        break

        def left_selector():
            pivot = 0
            for i in range(len(leftparts)):
                for j in range(pivot, len(rightparts)):
                    if leftuniques[i] & rightuniques[j]:
                        yield leftparts[i], rightparts[j]
                        pivot = j + 1
                        break
                else:
                    yield leftparts[i], None

        selector = {"left": left_selector, "inner": inner_selector}[how]

        rhs_dtypes = [(k, other._meta.dtypes[k]) for k in other._meta.columns]

        @delayed
        def fix_column(lhs):
            df = cudf.DataFrame()
            for k in lhs.columns:
                df[k + lsuffix] = lhs[k]

            for k, dtype in rhs_dtypes:
                data = np.zeros(len(lhs), dtype=dtype)
                mask_size = cudf.utils.utils.calc_chunk_size(
                    data.size, cudf.utils.utils.mask_bitsize
                )
                mask = np.zeros(mask_size, dtype=cudf.utils.utils.mask_dtype)
                sr = cudf.Series.from_masked_array(
                    data=data, mask=mask, null_count=data.size
                )

                df[k + rsuffix] = sr.set_index(df.index)

            return df

        joinedparts = [
            (
                part_join(lhs, rhs, how=how)
                if rhs is not None
                else fix_column(lhs)
            )
            for lhs, rhs in selector()
        ]

        meta = self._meta.join(
            other._meta, how=how, lsuffix=lsuffix, rsuffix=rsuffix
        )
        return from_delayed(joinedparts, meta=meta)

    def _align_divisions(self):
        """Align so that the values do not split across partitions
        """
        parts = self.to_delayed()
        uniques = self._get_unique_indices(parts=parts)
        originals = list(map(frozenset, uniques))

        changed = True
        while changed:
            changed = False
            for i in range(len(uniques))[:-1]:
                intersect = uniques[i] & uniques[i + 1]
                if intersect:
                    smaller = min(uniques[i], uniques[i + 1], key=len)
                    bigger = max(uniques[i], uniques[i + 1], key=len)
                    smaller |= intersect
                    bigger -= intersect
                    changed = True

        # Fix empty partitions
        uniques = list(filter(bool, uniques))

        return self._align_to_indices(
            uniques, originals=originals, parts=parts
        )

    def _get_unique_indices(self, parts=None):
        if parts is None:
            parts = self.to_delayed()

        @delayed
        def unique(x):
            return set(x.index.as_column().unique().to_array())

        parts = self.to_delayed()
        return compute(*map(unique, parts))

    def _align_to_indices(self, uniques, originals=None, parts=None):
        uniques = list(map(set, uniques))

        if parts is None:
            parts = self.to_delayed()

        if originals is None:
            originals = self._get_unique_indices(parts=parts)
            allindices = set()
            for x in originals:
                allindices |= x
            for us in uniques:
                us &= allindices
            uniques = list(filter(bool, uniques))

        extras = originals[-1] - uniques[-1]
        extras = {x for x in extras if x > max(uniques[-1])}

        if extras:
            uniques.append(extras)

        remap = OrderedDict()
        for idxset in uniques:
            remap[tuple(sorted(idxset))] = bins = []
            for i, orig in enumerate(originals):
                if idxset & orig:
                    bins.append(parts[i])

        @delayed
        def take(indices, depends):
            first = min(indices)
            last = max(indices)
            others = []
            for d in depends:
                # TODO: this can be replaced with searchsorted
                # Normalize to index data in range before selection.
                firstindex = d.index[0]
                lastindex = d.index[-1]
                s = max(first, firstindex)
                e = min(last, lastindex)
                others.append(d.loc[s:e])
            return cudf.concat(others)

        newparts = []
        for idx, depends in remap.items():
            newparts.append(take(idx, depends))

        divisions = list(map(min, uniques))
        divisions.append(max(uniques[-1]))

        newdd = from_delayed(newparts, meta=self._meta)
        return newdd, uniques

    def _compute_divisions(self):
        if self.known_divisions:
            return self

        @delayed
        def first_index(df):
            return df.index[0]

        @delayed
        def last_index(df):
            return df.index[-1]

        parts = self.to_delayed()
        divs = [first_index(p) for p in parts] + [last_index(parts[-1])]
        divisions = compute(*divs)
        return type(self)(self.dask, self._name, self._meta, divisions)

    def set_index(self, index, drop=True, sorted=False):
        """Set new index.

        Parameters
        ----------
        index : str or Series
            If a ``str`` is provided, it is used as the name of the
            column to be made into the index.
            If a ``Series`` is provided, it is used as the new index
        drop : bool
            Whether the first original index column is dropped.
        sorted : bool
            Whether the new index column is already sorted.
        """
        if not drop:
            raise NotImplementedError("drop=False not supported yet")

        if isinstance(index, str):
            tmpdf = self.sort_values(index)
            return tmpdf._set_column_as_sorted_index(index, drop=drop)
        elif isinstance(index, Series):
            indexname = "__dask_cudf.index"
            df = self.assign(**{indexname: index})
            return df.set_index(indexname, drop=drop, sorted=sorted)
        else:
            raise TypeError("cannot set_index from {}".format(type(index)))

    def _set_column_as_sorted_index(self, colname, drop):
        def select_index(df, col):
            return df.set_index(col)

        return self.map_partitions(
            select_index, col=colname, meta=self._meta.set_index(colname)
        )

    def _argsort(self, col, sorted=False):
        """
        Returns
        -------
        shufidx : Series
            Positional indices to be used with .take() to
            put the dataframe in order w.r.t ``col``.
        """
        # Get subset with just the index and positional value
        subset = self[col].to_dask_dataframe()
        subset = subset.reset_index(drop=False)
        ordered = subset.set_index(0, sorted=sorted)
        shufidx = from_dask_dataframe(ordered)["index"]
        return shufidx

    def _set_index_raw(self, indexname, drop, sorted):
        shufidx = self._argsort(indexname, sorted=sorted)
        # Shuffle the GPU data
        shuffled = self.take(shufidx, npartitions=self.npartitions)
        out = shuffled.map_partitions(lambda df: df.set_index(indexname))
        return out

    def reset_index(self, force=False, drop=False):
        """Reset index to range based
        """
        if force:
            dfs = self.to_delayed()
            sizes = np.asarray(compute(*map(delayed(len), dfs)))
            prefixes = np.zeros_like(sizes)
            prefixes[1:] = np.cumsum(sizes[:-1])

            @delayed
            def fix_index(df, startpos):
                stoppos = startpos + len(df)
                return df.set_index(
                    cudf.dataframe.RangeIndex(start=startpos, stop=stoppos)
                )

            outdfs = [
                fix_index(df, startpos) for df, startpos in zip(dfs, prefixes)
            ]
            return from_delayed(outdfs, meta=self._meta.reset_index(drop=True))
        else:
            return self.map_partitions(M.reset_index, drop=drop)

    def sort_values(self, by, ignore_index=False):
        """Sort by the given column

        Parameter
        ---------
        by : str
        """
        parts = self.to_delayed()
        sorted_parts = batcher_sortnet.sort_delayed_frame(parts, by)
        return from_delayed(sorted_parts, meta=self._meta).reset_index(
            force=not ignore_index
        )

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

    def _shuffle_sort_values(self, by):
        """Slow shuffle based sort by the given column

        Parameter
        ---------
        by : str
        """
        shufidx = self._argsort(by)
        return self.take(shufidx)

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
    outcol = cpp_reduce.apply_reduce("sum_of_squares", x)
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
    _partition_type = cudf.dataframe.index.Index


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


from_cudf = dd.from_pandas


def from_dask_dataframe(df):
    return df.map_partitions(cudf.from_pandas)
