# Copyright (c) 2024, NVIDIA CORPORATION.

import functools

from dask_expr._cumulative import CumulativeBlockwise, TakeLast

##
## Custom expression classes
##


class CumulativeBlockwiseCudf(CumulativeBlockwise):
    @functools.cached_property
    def _args(self) -> list:
        return self.operands[:1]

    @functools.cached_property
    def _kwargs(self) -> dict:
        # Must pass axis and skipna as kwargs in cudf
        return {"axis": self.axis, "skipna": self.skipna}


CumulativeBlockwise._args = CumulativeBlockwiseCudf._args
CumulativeBlockwise._kwargs = CumulativeBlockwiseCudf._kwargs


class TakeLastCudf(TakeLast):
    @staticmethod
    def operation(a, skipna=True):
        if not len(a):
            return a
        if skipna:
            a = a.bfill()
        # Cannot use `squeeze` with cudf
        return a.tail(n=1).iloc[0]


TakeLast.operation = staticmethod(TakeLastCudf.operation)
