# Copyright (c) 2024, NVIDIA CORPORATION.

from dask_expr._cumulative import CumulativeBlockwise
from dask_expr._reductions import Var

##
## Custom expression patching
##


# This can be removed after cudf#15176 is addressed.
# See: https://github.com/rapidsai/cudf/issues/15176
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


# This patch accounts for differences between
# numpy and cupy behavior. It may make sense
# to move this logic upstream.
_dx_reduction_aggregate = Var.reduction_aggregate


def _reduction_aggregate(*args, **kwargs):
    result = _dx_reduction_aggregate(*args, **kwargs)
    if result.ndim == 0:
        # cupy will sometimes produce a 0d array, and
        # we need to convert it to a scalar.
        return result.item()
    return result


Var.reduction_aggregate = staticmethod(_reduction_aggregate)
