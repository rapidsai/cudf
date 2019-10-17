from dask.dataframe.core import get_parallel_type, make_meta, meta_nonempty
from dask.dataframe.methods import concat_dispatch

import cudf

from .core import DataFrame, Index, Series

get_parallel_type.register(cudf.DataFrame, lambda _: DataFrame)
get_parallel_type.register(cudf.Series, lambda _: Series)
get_parallel_type.register(cudf.Index, lambda _: Index)


@meta_nonempty.register((cudf.DataFrame, cudf.Series, cudf.Index))
def meta_nonempty_cudf(x, index=None):
    y = meta_nonempty(x.to_pandas())  # TODO: add iloc[:5]
    return cudf.from_pandas(y)


@make_meta.register((cudf.Series, cudf.DataFrame))
def make_meta_cudf(x, index=None):
    return x.head(0)


@make_meta.register(cudf.Index)
def make_meta_cudf_index(x, index=None):
    return x[:0]


@concat_dispatch.register((cudf.DataFrame, cudf.Series, cudf.Index))
def concat_cudf(
    dfs, axis=0, join="outer", uniform=False, filter_warning=True, sort=None
):
    assert axis == 0
    assert join == "outer"
    return cudf.concat(dfs)


try:

    from dask.dataframe.methods import group_split, hash_df

    @hash_df.register(cudf.DataFrame)
    def hash_df_cudf(dfs):
        return dfs.hash_columns()

    @hash_df.register(cudf.Index)
    def hash_df_cudf_index(ind):
        from cudf.core.column import column, numerical

        cols = [column.as_column(ind)]
        return cudf.Series(numerical.column_hash_values(*cols))

    @group_split.register(cudf.DataFrame)
    def group_split_cudf(df, c, k):
        return df.scatter_by_map(c, map_size=k)


except ImportError:
    pass
