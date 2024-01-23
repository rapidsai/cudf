# Copyright (c) 2024, NVIDIA CORPORATION.

from dask_expr._cumulative import CumulativeBlockwise, TakeLast

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
