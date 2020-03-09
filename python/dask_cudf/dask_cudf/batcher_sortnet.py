"""
Batcher's Odd-even sorting network
Adapted from https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
"""
import math
from operator import getitem

import numpy as np

from dask import compute, delayed
from dask.base import tokenize
from dask.dataframe.core import DataFrame, _concat
from dask.dataframe.shuffle import rearrange_by_column_tasks
from dask.dataframe.utils import group_split_dispatch, hash_object_dispatch
from dask.highlevelgraph import HighLevelGraph
from dask.utils import digit, insert

import cudf as gd

try:
    import cytoolz as toolz
except ImportError:
    import toolz


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


def set_partitions_hash(df, columns, npartitions):
    c = hash_object_dispatch(df[columns], index=False)
    return np.mod(c, npartitions)


def _shuffle_group(df, columns, stage, k, npartitions, ignore_index):
    c = hash_object_dispatch(df[columns], index=False)
    typ = np.min_scalar_type(npartitions * 2)
    c = np.mod(c, npartitions).astype(typ, copy=False)
    np.floor_divide(c, k ** stage, out=c)
    np.mod(c, k, out=c)
    return group_split_dispatch(
        df, c.astype(np.int32), k, ignore_index=ignore_index
    )


def rearrange_by_hash(
    df, columns, npartitions, max_branch=None, ignore_index=True
):
    if npartitions and npartitions != df.npartitions:
        # Use main-line dask for new npartitions
        meta = df._meta._constructor_sliced([0])
        partitions = df[columns].map_partitions(
            set_partitions_hash, columns, npartitions, meta=meta
        )
        # Note: Dask will use a shallow copy for assign
        df2 = df.assign(_partitions=partitions)
        return rearrange_by_column_tasks(
            df2,
            "_partitions",
            max_branch=max_branch,
            npartitions=npartitions,
            ignore_index=ignore_index,
        )

    n = df.npartitions
    if max_branch is False:
        stages = 1
    else:
        max_branch = max_branch or 32
        stages = int(math.ceil(math.log(n) / math.log(max_branch)))

    if stages > 1:
        k = int(math.ceil(n ** (1 / stages)))
    else:
        k = n

    if isinstance(columns, str):
        columns = [columns]
    elif isinstance(columns, tuple):
        columns = list(columns)

    groups = []
    splits = []
    combines = []

    inputs = [
        tuple(digit(i, j, k) for j in range(stages))
        for i in range(k ** stages)
    ]

    token = tokenize(df, columns, max_branch)

    start = {
        ("shuffle-combine-" + token, 0, inp): (df._name, i)
        if i < df.npartitions
        else df._meta
        for i, inp in enumerate(inputs)
    }

    for stage in range(1, stages + 1):
        group = {  # Convert partition into dict of dataframe pieces
            ("shuffle-group-" + token, stage, inp): (
                _shuffle_group,
                ("shuffle-combine-" + token, stage - 1, inp),
                columns,
                stage - 1,
                k,
                n,
                ignore_index,
            )
            for inp in inputs
        }

        split = {  # Get out each individual dataframe piece from the dicts
            ("shuffle-split-" + token, stage, i, inp): (
                getitem,
                ("shuffle-group-" + token, stage, inp),
                i,
            )
            for i in range(k)
            for inp in inputs
        }

        combine = {  # concatenate those pieces together, with their friends
            ("shuffle-combine-" + token, stage, inp): (
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
                ignore_index,
            )
            for inp in inputs
        }
        groups.append(group)
        splits.append(split)
        combines.append(combine)

    end = {
        ("shuffle-" + token, i): ("shuffle-combine-" + token, stages, inp)
        for i, inp in enumerate(inputs)
    }

    dsk = toolz.merge(start, end, *(groups + splits + combines))
    graph = HighLevelGraph.from_collections(
        "shuffle-" + token, dsk, dependencies=[df]
    )
    df2 = DataFrame(graph, "shuffle-" + token, df, df.divisions)
    df2.divisions = (None,) * (df.npartitions + 1)

    return df2
