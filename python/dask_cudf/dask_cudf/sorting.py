# Copyright (c) 2020, NVIDIA CORPORATION.
import math
import warnings
from collections.abc import Iterator
from operator import getitem

import cupy
import numpy as np
import toolz

from dask import compute, delayed
from dask.base import tokenize
from dask.dataframe.core import DataFrame, Index, Series, _concat
from dask.dataframe.shuffle import rearrange_by_column, shuffle_group_get
from dask.dataframe.utils import group_split_dispatch
from dask.highlevelgraph import HighLevelGraph
from dask.utils import M, digit, insert

import cudf as gd

try:
    from .explicit_shuffle import explicit_sorted_shuffle

    explicit_comms = True
except ImportError:
    explicit_comms = False


"""
Batcher's Odd-even sorting network
Adapted from https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
"""


def get_oversized(length):
    """
    The oddeven network requires a power-of-2 length.
    This method computes the next power-of-2 from the *length* if
    *length* is not a power-of-2 value.
    """
    return 2 ** math.ceil(math.log2(length))


def is_power_of_2(length):
    return math.log2(length).is_integer()


def oddeven_merge(lo, hi, r):
    step = r * 2
    if step < hi - lo:
        for each in oddeven_merge(lo, hi, step):
            yield each
        for each in oddeven_merge(lo + r, hi, step):
            yield each
        for i in range(lo + r, hi - r, step):
            yield (i, i + r)
    else:
        yield (lo, lo + r)


def oddeven_merge_sort_range(lo, hi):
    """ sort the part of x with indices between lo and hi.

    Note: endpoints (lo and hi) are included.
    """
    if (hi - lo) >= 1:
        # if there is more than one element, split the input
        # down the middle and first sort the first and second
        # half, followed by merging them.
        mid = lo + ((hi - lo) // 2)
        for each in oddeven_merge_sort_range(lo, mid):
            yield each
        for each in oddeven_merge_sort_range(mid + 1, hi):
            yield each
        for each in oddeven_merge(lo, hi, 1):
            yield each


def oddeven_merge_sort(length):
    """ "length" is the length of the list to be sorted.
    Returns a list of pairs of indices starting with 0 """
    assert is_power_of_2(length)
    for each in oddeven_merge_sort_range(0, length - 1):
        yield each


def _pad_data_to_length(parts):
    parts = list(parts)
    needed = get_oversized(len(parts))
    padn = needed - len(parts)
    return parts + [None] * padn, len(parts)


def _compare_frame(a, b, max_part_size, by):
    if a is not None and b is not None:
        joint = gd.concat([a, b])
        sorten = joint.sort_values(by=by)
        # Split the sorted frame using the *max_part_size*
        lhs, rhs = sorten[:max_part_size], sorten[max_part_size:]
        # Replace empty frame with None
        return lhs or None, rhs or None
    elif a is None and b is None:
        return None, None
    elif a is None:
        return b.sort_values(by=by), None
    else:
        return a.sort_values(by=by), None


def _compare_and_swap_frame(parts, a, b, max_part_size, by):
    compared = delayed(_compare_frame)(
        parts[a], parts[b], max_part_size, by=by
    )
    parts[a] = compared[0]
    parts[b] = compared[1]


def _cleanup(df):
    if "__dask_cudf__valid" in df.columns:
        out = df.query("__dask_cudf__valid")
        del out["__dask_cudf__valid"]
    else:
        out = df
    return out


def sort_delayed_frame(parts, by):
    """
    Parameters
    ----------
    parts :
        Delayed partitions of cudf.DataFrame
    by : str
        Column name by which to sort

    The sort will also rebalance the partition sizes so that all output
    partitions has partition size of atmost `max(original_partition_sizes)`.
    Therefore, they may be fewer partitions in the output.
    """
    # Empty frame?
    if len(parts) == 0:
        return parts
    # Compute maximum paritition size, which is needed
    # for non-uniform partition size
    max_part_size = delayed(max)(*map(delayed(len), parts))
    # Add empty partitions to match power-of-2 requirement.
    parts, valid = _pad_data_to_length(parts)
    # More than 1 input?
    if len(parts) > 1:
        # Build batcher's odd-even sorting network
        for a, b in oddeven_merge_sort(len(parts)):
            _compare_and_swap_frame(parts, a, b, max_part_size, by=by)
    # Single input?
    else:
        parts = [delayed(lambda x: x.sort_values(by=by))(parts[0])]
    # Count number of non-empty partitions
    valid_ct = delayed(sum)(
        list(map(delayed(lambda x: int(x is not None)), parts[:valid]))
    )
    valid = compute(valid_ct)[0]
    validparts = parts[:valid]
    return validparts


def set_partitions_pre(s, divisions):
    partitions = divisions.searchsorted(s, side="right") - 1

    # Use searchsorted to avoid string-compare limitations
    partitions[
        divisions.tail(1).searchsorted(s, side="right").astype("bool")
    ] = (len(divisions) - 2)

    return partitions


def sorted_split_divs(df, divisions, col, stage, k, npartitions, sort_by):
    # Get partitions
    dtype = df[col].dtype
    splits = df[col].searchsorted(
        df._constructor_sliced(divisions, dtype=dtype), side="left"
    )
    splits[-1] = len(df[col])
    partitions = splits.tolist()

    # Create splits
    split_dict = {
        i: df.iloc[partitions[i] : partitions[i + 1]].copy(deep=False)
        for i in range(len(divisions) - 1)
    }

    if k < npartitions:
        # Rearrange the splits (for now -- Need NEW algorithm to avoid this)
        # Note that we REALLY don't want to do this if we dont need to!!
        agg_dict = {i: [] for i in range(k)}
        for c in [int(k) for k in split_dict.keys()]:
            c_new = np.mod(np.floor_divide(c, k ** stage), k)
            if split_dict[c] is not None and len(split_dict[c]):
                agg_dict[c_new].append(split_dict[c].copy(deep=False))
        split_dict = {}
        for i in range(k):
            if len(agg_dict[i]):
                split_dict[i] = gd.merge_sorted(agg_dict[i], keys=sort_by)
            else:
                split_dict[i] = df.iloc[:0]
    return split_dict


def sorted_split_divs_2(df, divisions, col):
    if not len(df):
        return {}, df

    # Get partitions
    dtype = df[col].dtype
    splits = df[col].searchsorted(
        df._constructor_sliced(divisions, dtype=dtype), side="left"
    )
    splits[-1] = len(df[col])
    partitions = splits.tolist()

    # Create splits
    result2 = {
        i: df.iloc[partitions[i] : partitions[i + 1]].copy(deep=False)
        for i in range(len(divisions) - 1)
        if partitions[i] != partitions[i + 1]
    }
    return result2, df.iloc[:0]


def shuffle_group_divs(df, divisions, col, stage, k, npartitions, inp):
    dtype = df[col].dtype
    c = set_partitions_pre(
        df[col], divisions=df._constructor_sliced(divisions, dtype=dtype)
    )
    typ = np.min_scalar_type(npartitions * 2)
    c = np.mod(c, npartitions).astype(typ, copy=False)
    np.floor_divide(c, k ** stage, out=c)
    np.mod(c, k, out=c)
    return dict(
        zip(range(k), df.scatter_by_map(c.astype(np.int32), map_size=k))
    )


def shuffle_group_divs_2(df, divisions, col):
    if not len(df):
        return {}, df
    ind = set_partitions_pre(
        df[col], divisions=df._constructor_sliced(divisions)
    ).astype(np.int32)
    result2 = group_split_dispatch(df, ind.view(np.int32), len(divisions) - 1)
    return result2, df.iloc[:0]


def _concat_wrapper(df_list, sort_by):
    if sort_by:
        return gd.merge_sorted(df_list, keys=sort_by)
    else:
        df = _concat(df_list)
        if sort_by:
            return df.sort_values(sort_by)
        return df


def rearrange_by_division_list(
    df, column: str, divisions: list, max_branch=None, sort_by=None
):
    npartitions = len(divisions) - 1
    n = df.npartitions
    max_branch = max_branch or 32
    stages = int(math.ceil(math.log(n) / math.log(max_branch)))

    if stages > 1:
        k = int(math.ceil(n ** (1 / stages)))
    else:
        k = n

    groups = []
    splits = []
    joins = []

    inputs = [
        tuple(digit(i, j, k) for j in range(stages))
        for i in range(k ** stages)
    ]

    token = tokenize(df, column, max_branch)

    start = {
        ("shuffle-join-" + token, 0, inp): (df._name, i)
        if i < df.npartitions
        else df._meta
        for i, inp in enumerate(inputs)
    }

    if sort_by:
        _split_func_1 = sorted_split_divs
        _split_func_2 = sorted_split_divs_2
    else:
        _split_func_1 = shuffle_group_divs
        _split_func_2 = shuffle_group_divs_2

    for stage in range(1, stages + 1):
        group = {  # Convert partition into dict of dataframe pieces
            ("shuffle-group-divs-" + token, stage, inp): (
                _split_func_1,
                ("shuffle-join-" + token, stage - 1, inp),
                divisions,
                column,
                stage - 1,
                k,
                n,
                sort_by,  # Need this to rearrange splits (for now)
            )
            for inp in inputs
        }

        split = {  # Get out each individual dataframe piece from the dicts
            ("shuffle-split-" + token, stage, i, inp): (
                getitem,
                ("shuffle-group-divs-" + token, stage, inp),
                i,
            )
            for i in range(k)
            for inp in inputs
        }

        join = {  # concatenate those pieces together, with their friends
            ("shuffle-join-" + token, stage, inp): (
                _concat_wrapper,
                [
                    (
                        "shuffle-split-" + token,
                        stage,
                        inp[stage - 1],
                        insert(inp, stage - 1, j),
                    )
                    for j in range(k)
                ],
                sort_by,
            )
            for inp in inputs
        }
        groups.append(group)
        splits.append(split)
        joins.append(join)

    end = {
        ("shuffle-" + token, i): ("shuffle-join-" + token, stages, inp)
        for i, inp in enumerate(inputs)
    }

    dsk = toolz.merge(start, end, *(groups + splits + joins))
    graph = HighLevelGraph.from_collections(
        "shuffle-" + token, dsk, dependencies=[df]
    )
    df2 = DataFrame(graph, "shuffle-" + token, df, df.divisions)

    if npartitions != df.npartitions:
        parts = [i % df.npartitions for i in range(npartitions)]
        token = tokenize(df2, npartitions)

        dsk = {
            ("repartition-group-" + token, i): (
                _split_func_2,
                k,
                divisions,
                column,
            )
            for i, k in enumerate(df2.__dask_keys__())
        }
        for p in range(npartitions):
            dsk[("repartition-get-" + token, p)] = (
                shuffle_group_get,
                ("repartition-group-" + token, parts[p]),
                p,
            )

        graph2 = HighLevelGraph.from_collections(
            "repartition-get-" + token, dsk, dependencies=[df2]
        )
        df3 = DataFrame(
            graph2, "repartition-get-" + token, df2, [None] * (npartitions + 1)
        )
    else:
        df3 = df2
        df3.divisions = (None,) * (df.npartitions + 1)

    return df3


def _quantile(a, q, interpolation="linear"):
    n = len(a)
    if not len(a):
        return None, n
    return (a.quantiles(q.tolist()), n)


def merge_quantiles(finalq, qs, vals):
    """ Combine several quantile calculations of different data.
    [NOTE: Logic copied from dask.array merge_percentiles]
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
    combined_vals_counts = gd.merge_sorted(
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


def quantile(df, q):
    """Approximate quantiles of Series.
    Parameters
    ----------
    q : list/array of floats
        Iterable of numbers ranging from 0 to 100 for the desired quantiles
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
    meta = df._meta_nonempty.quantile(q=q)

    # Define final action (create df with quantiles as index)
    def finalize_tsk(tsk):
        return (final_type, tsk, q)

    return_type = DataFrame

    # pandas/cudf uses quantile in [0, 1]
    # numpy / cupy uses [0, 100]
    qs = np.asarray(q)
    token = tokenize(df, qs)

    if len(qs) == 0:
        name = "quantiles-" + token
        empty_index = gd.Index([], dtype=float)
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
    return return_type(graph, name2, meta, new_divisions)


def sort_values_experimental(
    df,
    by,
    ignore_index=False,
    explicit_client=None,
    max_branch=None,
    divisions=None,
    sorted_split=False,
    upsample=1.0,
    set_divisions=False,
):
    """ Experimental sort_values implementation.

    Sort by the given column name or list/tuple of column names.

    Parameter
    ---------
    by : list, tuple, str
    """
    npartitions = df.npartitions
    if isinstance(by, str):
        by = [by]
    elif isinstance(by, tuple):
        by = list(by)

    # Step 1 - Pre-sort each partition
    if sorted_split:
        df2 = df.map_partitions(M.sort_values, by)
    else:
        df2 = df

    # Check if we are using explicit comms
    use_explicit = explicit_comms and explicit_client
    if use_explicit:
        npartitions = len(explicit_client.cluster.workers)

    # Step 2 - Calculate new divisions (if necessary)
    if divisions is None or (
        use_explicit and len(divisions) != npartitions + 1
    ):
        # TODO: Use input divisions for use_explicit==True
        qn = np.linspace(0.0, 1.0, npartitions + 1).tolist()
        divisions = quantile(df2[by], qn).compute().drop_duplicates()
        columns = divisions.columns
        # TODO: Make sure divisions are correctly handled for
        # non-numerical datatypes..
        if len(columns) == 1 and df2[columns[0]].dtype != "object":
            dtype = df2[columns[0]].dtype
            divisions = divisions[columns[0]].astype(dtype).values
            if dtype in ("int", "float"):
                divisions = divisions + 1
                divisions[0] = 0
            divisions = sorted(divisions.tolist())
        else:
            for col in columns:
                dtype = df2[col].dtype
                divisions[col] = divisions[col].astype(dtype)
                if dtype in ("int", "float"):
                    divisions[col] += 1
                    divisions[col].iloc[0] = 0
                elif dtype == "object":
                    divisions[col].iloc[-1] = chr(
                        ord(divisions[col].iloc[-1][0]) + 1
                    )
                    divisions[col].iloc[0] = chr(0)

    # Step 3 - Perform repartitioning shuffle
    sort_by = None
    if sorted_split:
        sort_by = by
    if use_explicit and len(by) == 1:
        # TODO: Handle len(by) > 1
        warnings.warn("Using explicit comms - This is an advanced feature.")
        df3 = explicit_sorted_shuffle(
            df2, by[0], divisions, sort_by, explicit_client
        )
    elif sorted_split and len(by) == 1:
        # Need to pass around divisions
        # TODO: Handle len(by) > 1
        df3 = rearrange_by_division_list(
            df2, by[0], divisions, max_branch=max_branch, sort_by=sort_by
        )
    else:
        # Lets assign a new partitions column
        # (That is: Use main-line dask shuffle)
        # TODO: Handle len(by) > 1
        meta = df2._meta._constructor_sliced([0])
        if not isinstance(divisions, (gd.Series, gd.DataFrame)):
            dtype = df2[by[0]].dtype
            divisions = df2._meta._constructor_sliced(divisions, dtype=dtype)

        partitions = df2[by].map_partitions(
            set_partitions_pre, divisions=divisions, meta=meta
        )

        df2b = df2.assign(_partitions=partitions)
        df3 = rearrange_by_column(
            df2b,
            "_partitions",
            max_branch=max_branch,
            npartitions=len(divisions) - 1,
            shuffle="tasks",
        ).drop(columns=["_partitions"])
    df3.divisions = (None,) * (df3.npartitions + 1)

    # Step 4 - Return final sorted df
    if sorted_split:
        # Data should already be sorted
        df4 = df3
    else:
        df4 = df3.map_partitions(M.sort_values, by)

    if not isinstance(divisions, gd.DataFrame) and set_divisions:
        # Can't have multi-column divisions elsewhere in dask (yet)
        df4.divisions = tuple(divisions)
    return df4
