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
    assert join == "outer"
    return cudf.concat(dfs, axis=axis)


try:

    from dask.dataframe.utils import group_split_dispatch, hash_object_dispatch
    from cudf.core.column import column, CategoricalColumn, StringColumn

    def _string_safe_hash(df):
        frame = df.copy(deep=False)
        for col in frame.columns:
            if isinstance(frame[col]._column, StringColumn):
                out_col = column.column_empty(
                    len(frame), dtype="int32", masked=False
                )
                ptr = out_col.data.ptr
                frame[col]._column.data_array_view.hash(devptr=ptr)
                frame[col] = out_col
        return frame.hash_columns()

    @hash_object_dispatch.register(cudf.DataFrame)
    def hash_object_cudf(frame, index=True):
        if index:
            return _string_safe_hash(frame.reset_index())
        return _string_safe_hash(frame)

    @hash_object_dispatch.register(cudf.Index)
    def hash_object_cudf_index(ind, index=None):

        if isinstance(ind, cudf.MultiIndex):
            return _string_safe_hash(ind.to_frame(index=False))

        col = column.as_column(ind)
        if isinstance(col, StringColumn):
            col = col.as_numerical_column("int32")
        elif isinstance(col, CategoricalColumn):
            col = col.as_numerical
        return cudf.Series(col).hash_values()

    @group_split_dispatch.register(cudf.DataFrame)
    def group_split_cudf(df, c, k):
        return dict(zip(range(k), df.scatter_by_map(c, map_size=k)))


except ImportError:
    pass
