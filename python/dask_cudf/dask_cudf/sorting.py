# Copyright (c) 2020, NVIDIA CORPORATION.
import math
import warnings
from operator import getitem

import numpy as np
import toolz

from dask import compute, delayed
from dask.base import tokenize
from dask.dataframe.core import DataFrame, _concat
from dask.dataframe.shuffle import set_partitions_pre, shuffle_group_get
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


def shuffle_group_divs(df, divisions, col, stage, k, npartitions):
    dtype = df[col].dtype
    c = set_partitions_pre(
        df[col], divisions=df._constructor_sliced(divisions, dtype=dtype)
    )
    typ = np.min_scalar_type(npartitions * 2)
    c = np.mod(c, npartitions).astype(typ, copy=False)
    np.floor_divide(c, k ** stage, out=c)
    np.mod(c, k, out=c)
    return group_split_dispatch(df, c.astype(np.int64), k)


def shuffle_group_divs_2(df, divisions, col):
    if not len(df):
        return {}, df
    ind = set_partitions_pre(
        df[col], divisions=df._constructor_sliced(divisions)
    ).astype(np.int64)
    n = ind.max() + 1
    result2 = group_split_dispatch(df, ind.values.view(np.int64), n)
    return result2, df.iloc[:0]


def rearrange_by_divisions(df, column: str, divisions: list, max_branch=None):
    npartitions = len(divisions) - 1
    max_branch = max_branch or 32
    n = df.npartitions

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

    for stage in range(1, stages + 1):
        group = {  # Convert partition into dict of dataframe pieces
            ("shuffle-group-divs-" + token, stage, inp): (
                shuffle_group_divs,
                ("shuffle-join-" + token, stage - 1, inp),
                divisions,
                column,
                stage - 1,
                k,
                n,
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
                _concat,
                [
                    (
                        "shuffle-split-" + token,
                        stage,
                        inp[stage - 1],
                        insert(inp, stage - 1, j),
                    )
                    for j in range(k)
                ],
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
                shuffle_group_divs_2,
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


def sort_values_experimental(df, by, ignore_index=False, explicit_client=None):
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

    # Make sure first column is numeric
    # (Cannot handle string column here yet)
    if isinstance(df[by[0]]._meta._column, gd.core.column.string.StringColumn):
        return df.sort_values(
            by, ignore_index=ignore_index, experimental=False
        )

    # Step 1 - Pre-sort each partition
    # df2 = df.map_partitions(M.sort_values, by)
    df2 = df  # TODO: Use Presort when k-way merge/concat is avilable.

    # Only handle single-column partitioning (for now)
    #     TODO: Handle partitioning on multiple columns?
    if len(by) > 1:
        warnings.warn(
            "Using experimental version of sort_values."
            " Only `by[0]` will be used for partitioning."
        )
    index = by[0]

    # Check if we are using explicit comms
    use_explicit = explicit_comms and explicit_client
    if use_explicit:
        npartitions = len(explicit_client.cluster.workers)

    # Step 2 - Calculate new divisions
    divisions = (
        df2[index]
        ._repartition_quantiles(npartitions, upsample=1.0)
        .compute()
        .to_list()
    )

    # Step 3 - Perform repartitioning shuffle
    if use_explicit:
        warnings.warn("Using explicit comms - This is an advanced feature.")
        df3 = explicit_sorted_shuffle(
            df2, index, divisions, sort_by=by, sorted_split=False
        )
        df3.divisions = (None,) * (npartitions + 1)
        return df3
    else:
        df3 = rearrange_by_divisions(df2, index, divisions)
        df3.divisions = (None,) * (npartitions + 1)

        # Step 4 - Return final sorted df
        #          (Can remove after k-way merge added)
        return df3.map_partitions(M.sort_values, by)
