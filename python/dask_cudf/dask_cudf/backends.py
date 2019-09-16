import numpy as np

from dask.dataframe.core import get_parallel_type, make_meta, meta_nonempty
from dask.dataframe.methods import (
    concat_dispatch,
    group_split,
    group_split_2,
    hash_df,
    percentiles_summary,
)

import cudf
import cudf._lib as libcudf

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


@hash_df.register(cudf.DataFrame)
def hash_df_cudf(dfs):
    return dfs.hash_columns()


@hash_df.register(cudf.Index)
def hash_df_cudf_index(ind):
    from cudf.core.column import column, numerical

    cols = [column.as_column(ind)]
    return cudf.Series(numerical.column_hash_values(*cols))


@group_split.register((cudf.DataFrame, cudf.Series, cudf.Index))
def group_split_cudf(df, c, k):
    source = [df[col] for col in df.columns]
    ind_map = cudf.Series(c.astype(np.int64))
    # TODO: Use proper python API (#2807)
    tables = libcudf.copying.scatter_to_frames(source, ind_map)
    for i in range(k - len(tables)):
        tables.append(tables[0].iloc[[]])
    return dict(zip(range(k), tables))


@group_split_2.register((cudf.DataFrame, cudf.Series, cudf.Index))
def group_split_2_cudf(df, col):
    if not len(df):
        return {}, df
    source = [df[c] for c in df.columns]
    ind_map = df[col]
    n = ind_map.max() + 1
    # TODO: Use proper python API (#2807)
    parts = libcudf.copying.scatter_to_frames(source, ind_map)
    for i in range(n - len(parts)):
        parts.append(parts[0].iloc[[]])
    result2 = dict(zip(range(n), parts))
    return result2, df.iloc[:0]


@percentiles_summary.register(cudf.Series)
def percentiles_summary_cudf(df, num_old, num_new, upsample, state):
    from dask.dataframe.utils import is_categorical_dtype
    from dask.dataframe.partitionquantiles import (
        sample_percentiles,
        percentiles_to_weights,
    )
    from dask.array.percentile import _percentile

    length = len(df)
    if length == 0:
        return ()
    random_state = np.random.RandomState(state)
    qs = sample_percentiles(num_old, num_new, length, upsample, random_state)
    data = df.values
    interpolation = "linear"
    if is_categorical_dtype(data):
        data = data.codes
        interpolation = "nearest"
    vals, _ = _percentile(data, qs, interpolation=interpolation)
    if interpolation == "linear" and np.issubdtype(data.dtype, np.integer):
        vals = np.around(vals).astype(data.dtype)
    vals_and_weights = percentiles_to_weights(qs, vals, length)
    return vals_and_weights
