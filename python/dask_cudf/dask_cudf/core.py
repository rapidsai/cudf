# Copyright (c) 2018-2022, NVIDIA CORPORATION.

import math
import warnings
from distutils.version import LooseVersion

import numpy as np
import pandas as pd
from tlz import partition_all

import dask
import dask.dataframe.optimize
from dask import dataframe as dd
from dask.base import normalize_token, tokenize
from dask.dataframe.core import (
    Scalar,
    finalize,
    handle_out,
    make_meta as dask_make_meta,
    map_partitions,
)
from dask.dataframe.utils import raise_on_meta_error
from dask.highlevelgraph import HighLevelGraph
from dask.utils import M, OperatorMethodMixin, apply, derived_from, funcname

import cudf
from cudf import _lib as libcudf
from cudf.utils.utils import _dask_cudf_nvtx_annotate

from dask_cudf import sorting
from dask_cudf.accessors import ListMethods, StructMethods

DASK_VERSION = LooseVersion(dask.__version__)


class _Frame(dd.core._Frame, OperatorMethodMixin):
    """Superclass for DataFrame and Series

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

    def __dask_postcompute__(self):
        return finalize, ()

    def __dask_postpersist__(self):
        return type(self), (self._name, self._meta, self.divisions)

    @_dask_cudf_nvtx_annotate
    def __init__(self, dsk, name, meta, divisions):
        if not isinstance(dsk, HighLevelGraph):
            dsk = HighLevelGraph.from_collections(name, dsk, dependencies=[])
        self.dask = dsk
        self._name = name
        meta = dask_make_meta(meta)
        if not isinstance(meta, self._partition_type):
            raise TypeError(
                f"Expected meta to specify type "
                f"{self._partition_type.__name__}, got type "
                f"{type(meta).__name__}"
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

    @_dask_cudf_nvtx_annotate
    def to_dask_dataframe(self, **kwargs):
        """Create a dask.dataframe object from a dask_cudf object"""
        nullable_pd_dtype = kwargs.get("nullable_pd_dtype", False)
        return self.map_partitions(
            M.to_pandas, nullable_pd_dtype=nullable_pd_dtype
        )


concat = dd.concat


normalize_token.register(_Frame, lambda a: a._name)


class DataFrame(_Frame, dd.core.DataFrame):
    _partition_type = cudf.DataFrame

    @_dask_cudf_nvtx_annotate
    def _assign_column(self, k, v):
        def assigner(df, k, v):
            out = df.copy()
            out[k] = v
            return out

        meta = assigner(self._meta, k, dask_make_meta(v))
        return self.map_partitions(assigner, k, v, meta=meta)

    @_dask_cudf_nvtx_annotate
    def apply_rows(self, func, incols, outcols, kwargs=None, cache_key=None):
        import uuid

        if kwargs is None:
            kwargs = {}

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

    @_dask_cudf_nvtx_annotate
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

    @_dask_cudf_nvtx_annotate
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

    @_dask_cudf_nvtx_annotate
    def set_index(self, other, sorted=False, divisions=None, **kwargs):
        if kwargs.pop("shuffle", "tasks") != "tasks":
            raise ValueError(
                "Dask-cudf only supports task based shuffling, got %s"
                % kwargs["shuffle"]
            )
        pre_sorted = sorted
        del sorted

        if (
            divisions == "quantile"
            or isinstance(divisions, (cudf.DataFrame, cudf.Series))
            or (
                isinstance(other, str)
                and cudf.api.types.is_string_dtype(self[other].dtype)
            )
        ):

            # Let upstream-dask handle "pre-sorted" case
            if pre_sorted:
                return dd.shuffle.set_sorted_index(
                    self, other, divisions=divisions, **kwargs
                )

            by = other
            if not isinstance(other, list):
                by = [by]
            if len(by) > 1:
                raise ValueError("Dask does not support MultiIndex (yet).")
            if divisions == "quantile":
                divisions = None

            # Use dask_cudf's sort_values
            # TODO: Handle `sorted=True`
            df = self.sort_values(
                by,
                max_branch=kwargs.get("max_branch", None),
                divisions=divisions,
                set_divisions=True,
                ignore_index=True,
            )

            # Ignore divisions if its a dataframe
            if isinstance(divisions, cudf.DataFrame):
                divisions = None

            # Set index and repartition
            df2 = df.map_partitions(
                sorting.set_index_post,
                index_name=other,
                drop=kwargs.get("drop", True),
                column_dtype=df.columns.dtype,
            )
            npartitions = kwargs.get("npartitions", self.npartitions)
            partition_size = kwargs.get("partition_size", None)
            if partition_size:
                return df2.repartition(partition_size=partition_size)
            if not divisions and df2.npartitions != npartitions:
                return df2.repartition(npartitions=npartitions)
            if divisions and df2.npartitions != len(divisions) - 1:
                return df2.repartition(divisions=divisions)
            return df2

        return super().set_index(
            other,
            sorted=pre_sorted,
            shuffle="tasks",
            divisions=divisions,
            **kwargs,
        )

    @_dask_cudf_nvtx_annotate
    def sort_values(
        self,
        by,
        ignore_index=False,
        max_branch=None,
        divisions=None,
        set_divisions=False,
        ascending=True,
        na_position="last",
        sort_function=None,
        sort_function_kwargs=None,
        **kwargs,
    ):
        if kwargs:
            raise ValueError(
                f"Unsupported input arguments passed : {list(kwargs.keys())}"
            )

        df = sorting.sort_values(
            self,
            by,
            max_branch=max_branch,
            divisions=divisions,
            set_divisions=set_divisions,
            ignore_index=ignore_index,
            ascending=ascending,
            na_position=na_position,
            sort_function=sort_function,
            sort_function_kwargs=sort_function_kwargs,
        )

        if ignore_index:
            return df.reset_index(drop=True)
        return df

    @_dask_cudf_nvtx_annotate
    def to_parquet(self, path, *args, **kwargs):
        """Calls dask.dataframe.io.to_parquet with CudfEngine backend"""
        from dask_cudf.io import to_parquet

        return to_parquet(self, path, *args, **kwargs)

    @_dask_cudf_nvtx_annotate
    def to_orc(self, path, **kwargs):
        """Calls dask_cudf.io.to_orc"""
        from dask_cudf.io import to_orc

        return to_orc(self, path, **kwargs)

    @derived_from(pd.DataFrame)
    @_dask_cudf_nvtx_annotate
    def var(
        self,
        axis=None,
        skipna=True,
        ddof=1,
        split_every=False,
        dtype=None,
        out=None,
        naive=False,
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
        elif naive:
            return _naive_var(self, meta, skipna, ddof, split_every, out)
        else:
            return _parallel_var(self, meta, skipna, split_every, out)

    @_dask_cudf_nvtx_annotate
    def repartition(self, *args, **kwargs):
        """Wraps dask.dataframe DataFrame.repartition method.
        Uses DataFrame.shuffle if `columns=` is specified.
        """
        # TODO: Remove this function in future(0.17 release)
        columns = kwargs.pop("columns", None)
        if columns:
            warnings.warn(
                "The columns argument will be removed from repartition in "
                "future versions of dask_cudf. Use DataFrame.shuffle().",
                FutureWarning,
            )
            warnings.warn(
                "Rearranging data by column hash. Divisions will lost. "
                "Set ignore_index=False to preserve Index values."
            )
            ignore_index = kwargs.pop("ignore_index", True)
            return self.shuffle(
                on=columns, ignore_index=ignore_index, **kwargs
            )
        return super().repartition(*args, **kwargs)

    @_dask_cudf_nvtx_annotate
    def shuffle(self, *args, **kwargs):
        """Wraps dask.dataframe DataFrame.shuffle method"""
        shuffle_arg = kwargs.pop("shuffle", None)
        if shuffle_arg and shuffle_arg != "tasks":
            raise ValueError("dask_cudf does not support disk-based shuffle.")
        return super().shuffle(*args, shuffle="tasks", **kwargs)

    @_dask_cudf_nvtx_annotate
    def groupby(self, by=None, **kwargs):
        from .groupby import CudfDataFrameGroupBy

        return CudfDataFrameGroupBy(self, by=by, **kwargs)


@_dask_cudf_nvtx_annotate
def sum_of_squares(x):
    x = x.astype("f8")._column
    outcol = libcudf.reduce.reduce("sum_of_squares", x)
    return cudf.Series(outcol)


@_dask_cudf_nvtx_annotate
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


@_dask_cudf_nvtx_annotate
def nlargest_agg(x, **kwargs):
    return cudf.concat(x).nlargest(**kwargs)


@_dask_cudf_nvtx_annotate
def nsmallest_agg(x, **kwargs):
    return cudf.concat(x).nsmallest(**kwargs)


class Series(_Frame, dd.core.Series):
    _partition_type = cudf.Series

    @_dask_cudf_nvtx_annotate
    def count(self, split_every=False):
        return reduction(
            [self],
            chunk=M.count,
            aggregate=np.sum,
            split_every=split_every,
            meta="i8",
        )

    @_dask_cudf_nvtx_annotate
    def mean(self, split_every=False):
        sum = self.sum(split_every=split_every)
        n = self.count(split_every=split_every)
        return sum / n

    @derived_from(pd.DataFrame)
    @_dask_cudf_nvtx_annotate
    def var(
        self,
        axis=None,
        skipna=True,
        ddof=1,
        split_every=False,
        dtype=None,
        out=None,
        naive=False,
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
        elif naive:
            return _naive_var(self, meta, skipna, ddof, split_every, out)
        else:
            return _parallel_var(self, meta, skipna, split_every, out)

    @_dask_cudf_nvtx_annotate
    def groupby(self, *args, **kwargs):
        from .groupby import CudfSeriesGroupBy

        return CudfSeriesGroupBy(self, *args, **kwargs)

    @property  # type: ignore
    @_dask_cudf_nvtx_annotate
    def list(self):
        return ListMethods(self)

    @property  # type: ignore
    @_dask_cudf_nvtx_annotate
    def struct(self):
        return StructMethods(self)


class Index(Series, dd.core.Index):
    _partition_type = cudf.Index  # type: ignore


@_dask_cudf_nvtx_annotate
def _naive_var(ddf, meta, skipna, ddof, split_every, out):
    num = ddf._get_numeric_data()
    x = 1.0 * num.sum(skipna=skipna, split_every=split_every)
    x2 = 1.0 * (num**2).sum(skipna=skipna, split_every=split_every)
    n = num.count(split_every=split_every)
    name = ddf._token_prefix + "var"
    result = map_partitions(
        var_aggregate, x2, x, n, token=name, meta=meta, ddof=ddof
    )
    if isinstance(ddf, DataFrame):
        result.divisions = (min(ddf.columns), max(ddf.columns))
    return handle_out(out, result)


@_dask_cudf_nvtx_annotate
def _parallel_var(ddf, meta, skipna, split_every, out):
    def _local_var(x, skipna):
        if skipna:
            n = x.count(skipna=skipna)
            avg = x.mean(skipna=skipna)
        else:
            # Not skipping nulls, so might as well
            # avoid the full `count` operation
            n = len(x)
            avg = x.sum(skipna=skipna) / n
        m2 = ((x - avg) ** 2).sum(skipna=skipna)
        return n, avg, m2

    def _aggregate_var(parts):
        n, avg, m2 = parts[0]
        for i in range(1, len(parts)):
            n_a, avg_a, m2_a = n, avg, m2
            n_b, avg_b, m2_b = parts[i]
            n = n_a + n_b
            avg = (n_a * avg_a + n_b * avg_b) / n
            delta = avg_b - avg_a
            m2 = m2_a + m2_b + delta**2 * n_a * n_b / n
        return n, avg, m2

    def _finalize_var(vals):
        n, _, m2 = vals
        return m2 / (n - 1)

    # Build graph
    nparts = ddf.npartitions
    if not split_every:
        split_every = nparts
    name = "var-" + tokenize(skipna, split_every, out)
    local_name = "local-" + name
    num = ddf._get_numeric_data()
    dsk = {
        (local_name, n, 0): (_local_var, (num._name, n), skipna)
        for n in range(nparts)
    }

    # Use reduction tree
    widths = [nparts]
    while nparts > 1:
        nparts = math.ceil(nparts / split_every)
        widths.append(nparts)
    height = len(widths)
    for depth in range(1, height):
        for group in range(widths[depth]):
            p_max = widths[depth - 1]
            lstart = split_every * group
            lstop = min(lstart + split_every, p_max)
            node_list = [
                (local_name, p, depth - 1) for p in range(lstart, lstop)
            ]
            dsk[(local_name, group, depth)] = (_aggregate_var, node_list)
    if height == 1:
        group = depth = 0
    dsk[(name, 0)] = (_finalize_var, (local_name, group, depth))

    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[num, ddf])
    result = dd.core.new_dd_object(graph, name, meta, (None, None))
    if isinstance(ddf, DataFrame):
        result.divisions = (min(ddf.columns), max(ddf.columns))
    return handle_out(out, result)


@_dask_cudf_nvtx_annotate
def _extract_meta(x):
    """
    Extract internal cache data (``_meta``) from dask_cudf objects
    """
    if isinstance(x, (Scalar, _Frame)):
        return x._meta
    elif isinstance(x, list):
        return [_extract_meta(_x) for _x in x]
    elif isinstance(x, tuple):
        return tuple(_extract_meta(_x) for _x in x)
    elif isinstance(x, dict):
        return {k: _extract_meta(v) for k, v in x.items()}
    return x


@_dask_cudf_nvtx_annotate
def _emulate(func, *args, **kwargs):
    """
    Apply a function using args / kwargs. If arguments contain dd.DataFrame /
    dd.Series, using internal cache (``_meta``) for calculation
    """
    with raise_on_meta_error(funcname(func)):
        return func(*_extract_meta(args), **_extract_meta(kwargs))


@_dask_cudf_nvtx_annotate
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


@_dask_cudf_nvtx_annotate
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

    npartitions = {arg.npartitions for arg in args if isinstance(arg, _Frame)}
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
    a = f"{token or funcname(chunk)}-chunk-{token_key}"
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
    b = f"{token or funcname(combine)}-combine-{token_key}"
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
    b = f"{token or funcname(aggregate)}-agg-{token_key}"
    conc = (list, [(a, depth, i) for i in range(k)])
    if aggregate_kwargs:
        dsk[(b, 0)] = (apply, aggregate, [conc], aggregate_kwargs)
    else:
        dsk[(b, 0)] = (aggregate, conc)

    if meta is None:
        meta_chunk = _emulate(apply, chunk, args, chunk_kwargs)
        meta = _emulate(apply, aggregate, [[meta_chunk]], aggregate_kwargs)
    meta = dask_make_meta(meta)

    graph = HighLevelGraph.from_collections(b, dsk, dependencies=args)
    return dd.core.new_dd_object(graph, b, meta, (None, None))


@_dask_cudf_nvtx_annotate
def from_cudf(data, npartitions=None, chunksize=None, sort=True, name=None):
    if isinstance(getattr(data, "index", None), cudf.MultiIndex):
        raise NotImplementedError(
            "dask_cudf does not support MultiIndex Dataframes."
        )

    name = name or ("from_cudf-" + tokenize(data, npartitions or chunksize))
    return dd.from_pandas(
        data,
        npartitions=npartitions,
        chunksize=chunksize,
        sort=sort,
        name=name,
    )


from_cudf.__doc__ = (
    "Wraps main-line Dask from_pandas...\n" + dd.from_pandas.__doc__
)


@_dask_cudf_nvtx_annotate
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
    kwargs = {"original": cudf.DataFrame} if DASK_VERSION >= "2.11.1" else {}
    DataFrame._bind_operator_method(name, meth, **kwargs)

    meth = getattr(cudf.Series, name)
    kwargs = {"original": cudf.Series} if DASK_VERSION >= "2.11.1" else {}
    Series._bind_operator_method(name, meth, **kwargs)

for name in ["lt", "gt", "le", "ge", "ne", "eq"]:
    meth = getattr(cudf.Series, name)
    kwargs = {"original": cudf.Series} if DASK_VERSION >= "2.11.1" else {}
    Series._bind_comparison_method(name, meth, **kwargs)
