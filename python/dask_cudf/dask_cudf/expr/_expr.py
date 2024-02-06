# Copyright (c) 2024, NVIDIA CORPORATION.

from dask_expr._cumulative import CumulativeBlockwise, TakeLast
from dask_expr._shuffle import DiskShuffle

##
## Custom expression patching
##


class PatchCumulativeBlockwise(CumulativeBlockwise):
    @property
    def _args(self) -> list:
        return self.operands[:1]

    @property
    def _kwargs(self) -> dict:
        # Must pass axis and skipna as kwargs in cudf
        return {"axis": self.axis, "skipna": self.skipna}


CumulativeBlockwise._args = PatchCumulativeBlockwise._args
CumulativeBlockwise._kwargs = PatchCumulativeBlockwise._kwargs


def _takelast(a, skipna=True):
    if not len(a):
        return a
    if skipna:
        a = a.bfill()
    # Cannot use `squeeze` with cudf
    return a.tail(n=1).iloc[0]


TakeLast.operation = staticmethod(_takelast)


def _shuffle_group(df, col, _filter, p):
    from dask.dataframe.shuffle import ensure_cleanup_on_exception

    with ensure_cleanup_on_exception(p):
        g = df.groupby(col)
        if hasattr(g, "_grouped"):
            # Avoid `get_group` for cudf data.
            # See: https://github.com/rapidsai/cudf/issues/14955
            keys, part_offsets, _, grouped_df = df.groupby(col)._grouped()
            d = {
                k: grouped_df.iloc[part_offsets[i] : part_offsets[i + 1]]
                for i, k in enumerate(keys.to_pandas())
                if k in _filter
            }
        else:
            d = {i: g.get_group(i) for i in g.groups if i in _filter}
        p.append(d, fsync=True)


DiskShuffle._shuffle_group = staticmethod(_shuffle_group)
