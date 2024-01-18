# Copyright (c) 2024, NVIDIA CORPORATION.

from dask_expr._groupby import GroupBy, SeriesGroupBy


class GroupByCudf(GroupBy):
    def __init__(self, *args, observed=None, **kwargs):
        observed = observed if observed is not None else True
        super().__init__(*args, observed=observed, **kwargs)


class SeriesGroupByCudf(SeriesGroupBy):
    def __init__(self, *args, observed=None, **kwargs):
        observed = observed if observed is not None else True
        super().__init__(*args, observed=observed, **kwargs)
