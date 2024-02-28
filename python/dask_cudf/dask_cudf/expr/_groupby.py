# Copyright (c) 2024, NVIDIA CORPORATION.

from dask_expr._groupby import (
    GroupBy as DXGroupBy,
    SeriesGroupBy as DXSeriesGroupBy,
)
from dask_expr._util import is_scalar

##
## Custom groupby classes
##

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


class SeriesGroupBy(DXSeriesGroupBy):
    def __init__(self, *args, observed=None, **kwargs):
        observed = observed if observed is not None else True
        super().__init__(*args, observed=observed, **kwargs)
