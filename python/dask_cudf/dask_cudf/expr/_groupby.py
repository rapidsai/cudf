# Copyright (c) 2024, NVIDIA CORPORATION.

from dask_expr._groupby import (
    GroupBy as DXGroupBy,
    SeriesGroupBy as DXSeriesGroupBy,
    SingleAggregation,
)
from dask_expr._util import is_scalar

from dask.dataframe.groupby import Aggregation

from cudf.core.groupby.groupby import _deprecate_collect

##
## Custom groupby classes
##


class ListAgg(SingleAggregation):
    @staticmethod
    def groupby_chunk(arg):
        return arg.agg(list)

    @staticmethod
    def groupby_aggregate(arg):
        gb = arg.agg(list)
        if gb.ndim > 1:
            for col in gb.columns:
                gb[col] = gb[col].list.concat()
            return gb
        else:
            return gb.list.concat()


list_aggregation = Aggregation(
    name="list",
    chunk=ListAgg.groupby_chunk,
    agg=ListAgg.groupby_aggregate,
)


def _translate_arg(arg):
    # Helper function to translate args so that
    # they can be processed correctly by upstream
    # dask & dask-expr. Right now, the only necessary
    # translation is list aggregations.
    if isinstance(arg, dict):
        return {k: _translate_arg(v) for k, v in arg.items()}
    elif isinstance(arg, list):
        return [_translate_arg(x) for x in arg]
    elif arg in ("collect", "list", list):
        return list_aggregation
    else:
        return arg


# TODO: These classes are mostly a work-around for missing
# `observed=False` support.
# See: https://github.com/rapidsai/cudf/issues/15173


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

    def collect(self, **kwargs):
        _deprecate_collect()
        return self._single_agg(ListAgg, **kwargs)

    def aggregate(self, arg, **kwargs):
        return super().aggregate(_translate_arg(arg), **kwargs)


class SeriesGroupBy(DXSeriesGroupBy):
    def __init__(self, *args, observed=None, **kwargs):
        observed = observed if observed is not None else True
        super().__init__(*args, observed=observed, **kwargs)

    def collect(self, **kwargs):
        _deprecate_collect()
        return self._single_agg(ListAgg, **kwargs)

    def aggregate(self, arg, **kwargs):
        return super().aggregate(_translate_arg(arg), **kwargs)
