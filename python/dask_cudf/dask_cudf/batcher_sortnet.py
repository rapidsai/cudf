"""
Batcher's Odd-even sorting network
Adapted from https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
"""
import math

from dask import compute, delayed

import cudf as gd


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
