import dask.dataframe as dd
from dask import delayed

import cudf


@delayed
def local_shuffle(frame, num_new_parts, key_columns):
    """Regroup the frame based on the key column(s)
    """
    partitions = frame.partition_by_hash(
        columns=key_columns, nparts=num_new_parts
    )
    return dict(enumerate(partitions))


@delayed
def get_subgroup(groups, i):
    out = groups.get(i)
    if out is None:
        return ()
    return out


def group_frame(frame_partitions, num_new_parts, key_columns):
    """Group frame to prepare for the join
    """
    return [
        local_shuffle(part, num_new_parts, key_columns)
        for part in frame_partitions
    ]


def fanout_subgroups(grouped_parts, num_new_parts):
    return [
        [get_subgroup(part, j) for part in grouped_parts]
        for j in range(num_new_parts)
    ]


def join_frames(left, right, on, how, lsuffix, rsuffix):
    """Join two frames on 1 or more columns.

    Parameters
    ----------
    left, right : dask_cudf.DataFrame
    on : tuple[str]
        key column(s)
    how : str
        Join method
    lsuffix, rsuffix : str
    """

    if on:
        on = [on] if isinstance(on, str) else list(on)

    empty_frame = left._meta.merge(
        right._meta, on=on, how=how, suffixes=(lsuffix, rsuffix)
    )

    def merge(left, right):
        return left.merge(right, on=on, how=how, suffixes=(lsuffix, rsuffix))

    left_val_names = [k for k in left.columns if k not in on]
    right_val_names = [k for k in right.columns if k not in on]
    same_names = set(left_val_names) & set(right_val_names)
    if same_names and not (lsuffix or rsuffix):
        raise ValueError(
            "there are overlapping columns but "
            "lsuffix and rsuffix are not defined"
        )

    dtypes = {k: left[k].dtype for k in left.columns}
    dtypes.update({k: right[k].dtype for k in right.columns})

    left_parts = left.to_delayed()
    right_parts = right.to_delayed()

    # Add column w/ hash(v) % nparts
    nparts = max(len(left_parts), len(right_parts))

    left_hashed = group_frame(left_parts, nparts, on)
    right_hashed = group_frame(right_parts, nparts, on)

    # Fanout each partition into nparts subgroups
    left_subgroups = fanout_subgroups(left_hashed, nparts)
    right_subgroups = fanout_subgroups(right_hashed, nparts)

    assert len(left_subgroups) == len(right_subgroups)

    # Concat
    left_cats = [delayed(cudf.concat, pure=True)(it) for it in left_subgroups]
    right_cats = [
        delayed(cudf.concat, pure=True)(it) for it in right_subgroups
    ]

    # Combine
    merged = [
        delayed(merge, pure=True)(left_cats[i], right_cats[i])
        for i in range(nparts)
    ]

    return dd.from_delayed(merged, prefix="join_result", meta=empty_frame)
