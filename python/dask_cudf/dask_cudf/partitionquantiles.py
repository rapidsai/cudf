# Copied from dask.dataframe
# [for temnporary dev purposes only]

import math
from collections.abc import Iterator
from functools import wraps

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64tz_dtype
from toolz import merge, merge_sorted, take

from dask.base import tokenize
from dask.dataframe.core import Series
from dask.dataframe.utils import is_categorical_dtype
from dask.utils import random_state_data


@wraps(np.percentile)
def _percentile(a, q, interpolation="linear"):
    n = len(a)
    if not len(a):
        return None, n
    if isinstance(q, Iterator):
        q = list(q)
    if a.dtype.name == "category":
        result = np.percentile(a.codes, q, interpolation=interpolation)
        import pandas as pd

        return pd.Categorical.from_codes(result, a.categories, a.ordered), n
    if np.issubdtype(a.dtype, np.datetime64):
        a2 = a.astype("i8")
        result = np.percentile(a2, q, interpolation=interpolation)
        return result.astype(a.dtype), n
    if not np.issubdtype(a.dtype, np.number):
        interpolation = "nearest"
    return np.percentile(a, q, interpolation=interpolation), n


def sample_percentiles(
    num_old, num_new, chunk_length, upsample=1.0, random_state=None
):
    """Construct percentiles for a chunk for repartitioning.

    Adapt the number of total percentiles calculated based on the number
    of current and new partitions.  Returned percentiles include equally
    spaced percentiles between [0, 100], and random percentiles.  See
    detailed discussion below.

    Parameters
    ----------
    num_old: int
        Number of partitions of the current object
    num_new: int
        Number of partitions of the new object
    chunk_length: int
        Number of rows of the partition
    upsample : float
        Multiplicative factor to increase the number of samples

    Returns
    -------
    qs : numpy.ndarray of sorted percentiles between 0, 100

    Constructing ordered (i.e., not hashed) partitions is hard.  Calculating
    approximate percentiles for generic objects in an out-of-core fashion is
    also hard.  Fortunately, partition boundaries don't need to be perfect
    in order for partitioning to be effective, so we strive for a "good enough"
    method that can scale to many partitions and is reasonably well-behaved for
    a wide variety of scenarios.

    Two similar approaches come to mind: (1) take a subsample of every
    partition, then find the best new partitions for the combined subsamples;
    and (2) calculate equally-spaced percentiles on every partition (a
    relatively cheap operation), then merge the results.  We do both, but
    instead of random samples, we use random percentiles.

    If the number of partitions isn't changing, then the ratio of fixed
    percentiles to random percentiles is 2 to 1.  If repartitioning goes from
    a very high number of partitions to a very low number of partitions, then
    we use more random percentiles, because a stochastic approach will be more
    stable to potential correlations in the data that may cause a few equally-
    spaced partitions to under-sample the data.

    The more partitions there are, then the more total percentiles will get
    calculated across all partitions.  Squaring the number of partitions
    approximately doubles the number of total percentiles calculated, so
    num_total_percentiles ~ sqrt(num_partitions).  We assume each partition
    is approximately the same length.  This should provide adequate resolution
    and allow the number of partitions to scale.

    For numeric data, one could instead use T-Digest for floats and Q-Digest
    for ints to calculate approximate percentiles.  Our current method works
    for any dtype.
    """
    # *waves hands*
    random_percentage = 1 / (1 + (4 * num_new / num_old) ** 0.5)
    num_percentiles = upsample * num_new * (num_old + 22) ** 0.55 / num_old
    num_fixed = int(num_percentiles * (1 - random_percentage)) + 2
    num_random = int(num_percentiles * random_percentage) + 2

    if num_fixed + num_random + 5 >= chunk_length:
        return np.linspace(0, 100, chunk_length + 1)

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    q_fixed = np.linspace(0, 100, num_fixed)
    q_random = random_state.rand(num_random) * 100
    q_edges = [60 / (num_fixed - 1), 100 - 60 / (num_fixed - 1)]
    qs = np.concatenate([q_fixed, q_random, q_edges, [0, 100]])
    qs.sort()
    # Make the divisions between percentiles a little more even
    qs = 0.5 * (qs[:-1] + qs[1:])
    return qs


def tree_width(N, to_binary=False):
    """Generate tree width suitable for ``merge_sorted`` given N inputs

    The larger N is, the more tasks are reduced in a single task.

    In theory, this is designed so all tasks are of comparable effort.
    """
    if N < 32:
        group_size = 2
    else:
        group_size = int(math.log(N))
    num_groups = N // group_size
    if to_binary or num_groups < 16:
        return 2 ** int(math.log(N / group_size, 2))
    else:
        return num_groups


def tree_groups(N, num_groups):
    """Split an integer N into evenly sized and spaced groups.

    >>> tree_groups(16, 6)
    [3, 2, 3, 3, 2, 3]
    """
    # Bresenham, you so smooth!
    group_size = N // num_groups
    dx = num_groups
    dy = N - group_size * num_groups
    D = 2 * dy - dx
    rv = []
    for _ in range(num_groups):
        if D < 0:
            rv.append(group_size)
        else:
            rv.append(group_size + 1)
            D -= 2 * dx
        D += 2 * dy
    return rv


def create_merge_tree(func, keys, token):
    """Create a task tree that merges all the keys with a reduction function.

    Parameters
    ----------
    func: callable
        Reduction function that accepts a single list of values to reduce.
    keys: iterable
        Keys to reduce from the source dask graph.
    token: object
        Included in each key of the returned dict.

    This creates a k-ary tree where k depends on the current level and is
    greater the further away a node is from the root node.  This reduces the
    total number of nodes (thereby reducing scheduler overhead), but still
    has beneficial properties of trees.

    For reasonable numbers of keys, N < 1e5, the total number of nodes in the
    tree is roughly ``N**0.78``.  For 1e5 < N < 2e5, is it roughly ``N**0.8``.
    """
    level = 0
    prev_width = len(keys)
    prev_keys = iter(keys)
    rv = {}
    while prev_width > 1:
        width = tree_width(prev_width)
        groups = tree_groups(prev_width, width)
        keys = [(token, level, i) for i in range(width)]
        rv.update(
            (key, (func, list(take(num, prev_keys))))
            for num, key in zip(groups, keys)
        )
        prev_width = width
        prev_keys = iter(keys)
        level += 1
    return rv


def percentiles_to_weights(qs, vals, length):
    """Weigh percentile values by length and the difference between percentiles

    >>> percentiles = np.array([0, 25, 50, 90, 100])
    >>> values = np.array([2, 3, 5, 8, 13])
    >>> length = 10
    >>> percentiles_to_weights(percentiles, values, length)
    ([2, 3, 5, 8, 13], [125.0, 250.0, 325.0, 250.0, 50.0])

    The weight of the first element, ``2``, is determined by the difference
    between the first and second percentiles, and then scaled by length:

    >>> 0.5 * length * (percentiles[1] - percentiles[0])
    125.0

    The second weight uses the difference of percentiles on both sides, so
    it will be twice the first weight if the percentiles are equally spaced:

    >>> 0.5 * length * (percentiles[2] - percentiles[0])
    250.0
    """
    if length == 0:
        return ()
    diff = np.ediff1d(qs, 0.0, 0.0)
    weights = 0.5 * length * (diff[1:] + diff[:-1])
    return vals.tolist(), weights.tolist()


def merge_and_compress_summaries(vals_and_weights):
    """Merge and sort percentile summaries that are already sorted.

    Each item is a tuple like ``(vals, weights)`` where vals and weights
    are lists.  We sort both by vals.

    Equal values will be combined, their weights summed together.
    """
    vals_and_weights = [x for x in vals_and_weights if x]
    if not vals_and_weights:
        return ()
    it = merge_sorted(*[zip(x, y) for x, y in vals_and_weights])
    vals = []
    weights = []
    vals_append = vals.append
    weights_append = weights.append
    val, weight = prev_val, prev_weight = next(it)
    for val, weight in it:
        if val == prev_val:
            prev_weight += weight
        else:
            vals_append(prev_val)
            weights_append(prev_weight)
            prev_val, prev_weight = val, weight
    if val == prev_val:
        vals_append(prev_val)
        weights_append(prev_weight)
    return vals, weights


def process_val_weights(vals_and_weights, npartitions, dtype_info):
    """Calculate final approximate percentiles given weighted vals

    ``vals_and_weights`` is assumed to be sorted.  We take a cumulative
    sum of the weights, which makes them percentile-like (their scale is
    [0, N] instead of [0, 100]).  Next we find the divisions to create
    partitions of approximately equal size.

    It is possible for adjacent values of the result to be the same.  Since
    these determine the divisions of the new partitions, some partitions
    may be empty.  This can happen if we under-sample the data, or if there
    aren't enough unique values in the column.  Increasing ``upsample``
    keyword argument in ``df.set_index`` may help.
    """
    dtype, info = dtype_info

    if not vals_and_weights:
        try:
            return np.array(None, dtype=dtype)
        except Exception:
            # dtype does not support None value so allow it to change
            return np.array(None, dtype=np.float_)

    vals, weights = vals_and_weights
    vals = np.array(vals)
    weights = np.array(weights)

    # We want to create exactly `npartition` number of groups of `vals` that
    # are approximately the same weight and non-empty if possible.  We use a
    # simple approach (more accurate algorithms exist):
    # 1. Remove all the values with weights larger than the relative
    #    percentile width from consideration (these are `jumbo`s)
    # 2. Calculate percentiles with "interpolation=left" of percentile-like
    #    weights of the remaining values.  These are guaranteed to be unique.
    # 3. Concatenate the values from (1) and (2), sort, and return.
    #
    # We assume that all values are unique, which happens in the previous
    # step `merge_and_compress_summaries`.

    if len(vals) == npartitions + 1:
        rv = vals
    elif len(vals) < npartitions + 1:
        # The data is under-sampled
        if np.issubdtype(vals.dtype, np.number) and not is_categorical_dtype(
            dtype
        ):
            # Interpolate extra divisions
            q_weights = np.cumsum(weights)
            q_target = np.linspace(
                q_weights[0], q_weights[-1], npartitions + 1
            )
            rv = np.interp(q_target, q_weights, vals)
        else:
            # Distribute the empty partitions
            duplicated_index = np.linspace(
                0, len(vals) - 1, npartitions - len(vals) + 1, dtype=int
            )
            duplicated_vals = vals[duplicated_index]
            rv = np.concatenate([vals, duplicated_vals])
            rv.sort()
    else:
        target_weight = weights.sum() / npartitions
        jumbo_mask = weights >= target_weight
        jumbo_vals = vals[jumbo_mask]

        trimmed_vals = vals[~jumbo_mask]
        trimmed_weights = weights[~jumbo_mask]
        trimmed_npartitions = npartitions - len(jumbo_vals)

        # percentile-like, but scaled by weights
        q_weights = np.cumsum(trimmed_weights)
        q_target = np.linspace(0, q_weights[-1], trimmed_npartitions + 1)

        left = np.searchsorted(q_weights, q_target, side="left")
        right = np.searchsorted(q_weights, q_target, side="right") - 1
        # stay inbounds
        np.maximum(right, 0, right)
        lower = np.minimum(left, right)
        trimmed = trimmed_vals[lower]

        rv = np.concatenate([trimmed, jumbo_vals])
        rv.sort()

    if is_categorical_dtype(dtype):
        rv = pd.Categorical.from_codes(rv, info[0], info[1])
    elif is_datetime64tz_dtype(dtype):
        rv = pd.DatetimeIndex(rv).tz_localize(dtype.tz)
    elif "datetime64" in str(dtype):
        rv = pd.DatetimeIndex(rv, dtype=dtype)
    elif rv.dtype != dtype:
        rv = rv.astype(dtype)
    return rv


def percentiles_summary(df, num_old, num_new, upsample, state):
    """Summarize data using percentiles and derived weights.

    These summaries can be merged, compressed, and converted back into
    approximate percentiles.

    Parameters
    ----------
    df: pandas.Series
        Data to summarize
    num_old: int
        Number of partitions of the current object
    num_new: int
        Number of partitions of the new object
    upsample: float
        Scale factor to increase the number of percentiles calculated in
        each partition.  Use to improve accuracy.
    """
    length = len(df)
    if length == 0:
        return ()
    random_state = np.random.RandomState(state)
    qs = sample_percentiles(num_old, num_new, length, upsample, random_state)
    data = df.values
    interpolation = "linear"
    if is_categorical_dtype(data):
        data = data.codes
        interpolation = "nearest"
    vals, n = _percentile(data, qs, interpolation=interpolation)
    if interpolation == "linear" and np.issubdtype(data.dtype, np.integer):
        vals = np.round(vals).astype(data.dtype)
    vals_and_weights = percentiles_to_weights(qs, vals, length)
    return vals_and_weights


def dtype_info(df):
    info = None
    if is_categorical_dtype(df):
        data = df.values
        info = (data.categories, data.ordered)
    return df.dtype, info


def partition_quantiles(df, npartitions, upsample=1.0, random_state=None):
    """ Approximate quantiles of Series used for repartitioning
    """
    assert isinstance(df, Series)
    # currently, only Series has quantile method
    # Index.quantile(list-like) must be pd.Series, not pd.Index
    return_type = Series

    qs = np.linspace(0, 1, npartitions + 1)
    token = tokenize(df, qs, upsample)
    if random_state is None:
        random_state = int(token, 16) % np.iinfo(np.int32).max
    state_data = random_state_data(df.npartitions, random_state)

    df_keys = df.__dask_keys__()

    name0 = "re-quantiles-0-" + token
    dtype_dsk = {(name0, 0): (dtype_info, df_keys[0])}

    name1 = "re-quantiles-1-" + token
    val_dsk = {
        (name1, i): (
            percentiles_summary,
            key,
            df.npartitions,
            npartitions,
            upsample,
            state,
        )
        for i, (state, key) in enumerate(zip(state_data, df_keys))
    }

    name2 = "re-quantiles-2-" + token
    merge_dsk = create_merge_tree(
        merge_and_compress_summaries, sorted(val_dsk), name2
    )
    if not merge_dsk:
        # Compress the data even if we only have one partition
        merge_dsk = {
            (name2, 0, 0): (merge_and_compress_summaries, [list(val_dsk)[0]])
        }

    merged_key = max(merge_dsk)

    name3 = "re-quantiles-3-" + token
    last_dsk = {
        (name3, 0): (
            pd.Series,  # TODO: Use `type(df._meta)` when cudf adds `tolist()`
            (process_val_weights, merged_key, npartitions, (name0, 0)),
            qs,
            None,
            df.name,
        )
    }

    dsk = merge(df.dask, dtype_dsk, val_dsk, merge_dsk, last_dsk)
    new_divisions = [0.0, 1.0]
    return return_type(dsk, name3, df._meta, new_divisions)
