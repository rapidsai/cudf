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

    from dask.dataframe.utils import group_split_dispatch, hash_object_dispatch

    @hash_object_dispatch.register(cudf.DataFrame)
    def hash_object_cudf(
        dfs, index=None, encoding=None, hash_key=None, categorize=None
    ):
        return dfs.hash_columns()

    @hash_object_dispatch.register(cudf.Index)
    def hash_object_cudf_index(
        ind, index=None, encoding=None, hash_key=None, categorize=None
    ):
        from cudf.core.column import column, numerical

        cols = [column.as_column(ind)]
        return cudf.Series(numerical.column_hash_values(*cols))

    @group_split_dispatch.register(cudf.DataFrame)
    def group_split_cudf(df, c, k):
        return df.scatter_by_map(c, map_size=k)


except ImportError:
    pass
