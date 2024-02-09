# Copyright (c) 2024, NVIDIA CORPORATION.

from functools import partial

from dask_expr._collection import new_collection
from dask_expr._groupby import (
    GroupBy as DXGroupBy,
    GroupbyAggregation,
    SeriesGroupBy as DXSeriesGroupBy,
)
from dask_expr._util import is_scalar

from dask_cudf.expr._expr import CudfGroupbyAggregation
from dask_cudf.groupby import OPTIMIZED_AGGS

##
## Custom groupby classes
##


class GroupBy(DXGroupBy):
    def __init__(self, *args, observed=None, **kwargs):
        observed = observed if observed is not None else True
        super().__init__(*args, observed=observed, **kwargs)
        # Add optimized aggregation code paths
        for agg in OPTIMIZED_AGGS:
            setattr(self, agg, partial(single_agg, self, agg))
        setattr(self, "agg", partial(groupby_agg, self))
        setattr(self, "aggregate", partial(groupby_agg, self))

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


class SeriesGroupBy(DXSeriesGroupBy):
    def __init__(self, *args, observed=None, **kwargs):
        observed = observed if observed is not None else True
        super().__init__(*args, observed=observed, **kwargs)
        # Add optimized aggregation code paths
        for agg in OPTIMIZED_AGGS:
            setattr(self, agg, partial(single_agg, self, agg))
        setattr(self, "agg", partial(groupby_agg, self))
        setattr(self, "aggregate", partial(groupby_agg, self))


def single_agg(gb, agg_name, **kwargs):
    _optimized = kwargs.pop("_optimized", agg_name == "collect")
    if _optimized and hasattr(gb.obj._meta, "to_pandas"):
        if gb._slice is None:
            if isinstance(gb.by, list):
                agg = {c: agg_name for c in gb.obj.columns if c not in gb.by}
            else:
                agg = {c: agg_name for c in gb.obj.columns if c != gb.by}
        else:
            agg = {gb._slice: agg_name}

        return groupby_agg(gb, agg, **kwargs)
    else:
        # Fall back to dask-expr code path
        return getattr(super(type(gb), gb), agg_name)(**kwargs)


def groupby_agg(
    gb,
    arg=None,
    split_every=8,
    split_out=1,
    shuffle_method=None,
    _optimized=True,
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
    if (
        _optimized
        and _aggs_optimized(arg, OPTIMIZED_AGGS)
        and hasattr(gb.obj._meta, "to_pandas")
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
