# Copyright (c) 2024, NVIDIA CORPORATION.

from dask_expr import DataFrame, FrameBase, Index, Series, get_collection_type

import cudf

##
## Custom collection classes
##


class DataFrameCudf(DataFrame):
    def groupby(
        self,
        by,
        group_keys=True,
        sort=None,
        observed=None,
        dropna=None,
        **kwargs,
    ):
        from dask_cudf.expr_backend._groupby import GroupByCudf

        if isinstance(by, FrameBase) and not isinstance(by, Series):
            raise ValueError(
                f"`by` must be a column name or list of columns, got {by}."
            )

        return GroupByCudf(
            self,
            by,
            group_keys=group_keys,
            sort=sort,
            observed=observed,
            dropna=dropna,
            **kwargs,
        )


class SeriesCudf(Series):
    def groupby(self, by, **kwargs):
        from dask_cudf.expr_backend._groupby import SeriesGroupByCudf

        return SeriesGroupByCudf(self, by, **kwargs)


class IndexCudf(Index):
    pass  # Same as pandas (for now)


get_collection_type.register(cudf.DataFrame, lambda _: DataFrameCudf)
get_collection_type.register(cudf.Series, lambda _: SeriesCudf)
get_collection_type.register(cudf.BaseIndex, lambda _: IndexCudf)
