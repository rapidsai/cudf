# Copyright (c) 2024, NVIDIA CORPORATION.

from dask_expr._collection import new_collection
from dask_expr._groupby import (
    GroupBy as DXGroupBy,
    GroupbyAggregation,
    SeriesGroupBy as DXSeriesGroupBy,
)
from dask_expr._util import is_scalar

from dask_cudf.expr._expr import CudfGroupbyAggregation

##
## Custom groupby classes
##


class GroupBy(DXGroupBy):
    def __init__(self, *args, observed=None, **kwargs):
        observed = observed if observed is not None else True
        super().__init__(*args, observed=observed, **kwargs)

    def __getitem__(self, key):
        if is_scalar(key):
            return SeriesGroupBy(
                self.obj,
                by=self.by,
                slice=key,
                sort=self.sort,
                dropna=self.dropna,
                observed=self.observed,
            )
        g = GroupBy(
            self.obj,
            by=self.by,
            slice=key,
            sort=self.sort,
            dropna=self.dropna,
            observed=self.observed,
            group_keys=self.group_keys,
        )
        return g

    def aggregate(self, *args, **kwargs):
        return _aggregation(self, *args, **kwargs)


class SeriesGroupBy(DXSeriesGroupBy):
    def __init__(self, *args, observed=None, **kwargs):
        observed = observed if observed is not None else True
        super().__init__(*args, observed=observed, **kwargs)

    def aggregate(self, *args, **kwargs):
        return _aggregation(self, *args, **kwargs)


def _aggregation(
    gb, arg=None, split_every=8, split_out=1, shuffle_method=None, **kwargs
):
    from dask_cudf.groupby import (
        OPTIMIZED_AGGS,
        _aggs_optimized,
        _redirect_aggs,
    )

    if arg is None:
        raise NotImplementedError("arg=None not supported")

    if arg == "size":
        return gb.size()

    arg = _redirect_aggs(arg)
    if _aggs_optimized(arg, OPTIMIZED_AGGS) and hasattr(
        gb.obj._meta, "to_pandas"
    ):
        cls = CudfGroupbyAggregation
    else:
        cls = GroupbyAggregation

    return new_collection(
        cls(
            gb.obj.expr,
            arg,
            gb.observed,
            gb.dropna,
            split_every,
            split_out,
            gb.sort,
            shuffle_method,
            gb._slice,
            *gb.by,
        )
    )
