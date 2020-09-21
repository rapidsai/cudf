import numpy as np

from dask.dataframe.core import get_parallel_type, make_meta, meta_nonempty
from dask.dataframe.methods import concat_dispatch

import cudf
from cudf.utils.dtypes import is_categorical_dtype, is_string_dtype

from .core import DataFrame, Index, Series

get_parallel_type.register(cudf.DataFrame, lambda _: DataFrame)
get_parallel_type.register(cudf.Series, lambda _: Series)
get_parallel_type.register(cudf.Index, lambda _: Index)


@meta_nonempty.register(cudf.Index)
def _nonempty_index(idx):
    if isinstance(idx, cudf.core.index.RangeIndex):
        return cudf.core.index.RangeIndex(2, name=idx.name)
    elif isinstance(idx, cudf.core.index.DatetimeIndex):
        start = "1970-01-01"
        data = np.array([start, "1970-01-02"], dtype=idx.dtype)
        values = cudf.core.column.as_column(data)
        return cudf.core.index.DatetimeIndex(values, name=idx.name)
    elif isinstance(idx, cudf.core.index.StringIndex):
        return cudf.core.index.StringIndex(["cat", "dog"], name=idx.name)
    elif isinstance(idx, cudf.core.index.CategoricalIndex):
        key = tuple(idx._data.keys())
        assert len(key) == 1
        categories = idx._data[key[0]].categories
        codes = [0, 0]
        ordered = idx._data[key[0]].ordered
        values = column.build_categorical_column(
            categories=categories, codes=codes, ordered=ordered
        )
        return cudf.core.index.CategoricalIndex(values, name=idx.name)
    elif isinstance(idx, cudf.core.index.GenericIndex):
        return cudf.core.index.GenericIndex(
            np.arange(2, dtype=idx.dtype), name=idx.name
        )
    elif isinstance(idx, cudf.core.MultiIndex):
        levels = [meta_nonempty(l) for l in idx.levels]
        codes = [[0, 0] for i in idx.levels]
        return cudf.core.MultiIndex(
            levels=levels, codes=codes, names=idx.names
        )

    raise TypeError(
        "Don't know how to handle index of type {0}".format(type(idx))
    )


@meta_nonempty.register(cudf.Series)
def _nonempty_series(s, idx=None):
    if idx is None:
        idx = _nonempty_index(s.index)
    dtype = s.dtype
    if is_categorical_dtype(dtype):
        categories = (
            s._column.categories if len(s._column.categories) else ["a"]
        )
        codes = [0, 0]
        ordered = s._column.ordered
        data = column.build_categorical_column(
            categories=categories, codes=codes, ordered=ordered
        )
    elif is_string_dtype(dtype):
        data = ["cat", "dog"]
    else:
        data = np.arange(start=0, stop=2, dtype=dtype)

    return cudf.Series(data, name=s.name, index=idx)


@meta_nonempty.register(cudf.DataFrame)
def meta_nonempty_cudf(x):
    idx = meta_nonempty(x.index)
    dt_s_dict = dict()
    data = dict()
    for i, c in enumerate(x.columns):
        series = x[c]
        dt = str(series.dtype)
        if dt not in dt_s_dict:
            dt_s_dict[dt] = _nonempty_series(series, idx=idx)
        data[i] = dt_s_dict[dt]
    res = cudf.DataFrame(data, index=idx, columns=np.arange(len(x.columns)))
    res.columns = x.columns
    return res


@make_meta.register((cudf.Series, cudf.DataFrame))
def make_meta_cudf(x, index=None):
    return x.head(0)


@make_meta.register(cudf.Index)
def make_meta_cudf_index(x, index=None):
    return x[:0]


@concat_dispatch.register((cudf.DataFrame, cudf.Series, cudf.Index))
def concat_cudf(
    dfs,
    axis=0,
    join="outer",
    uniform=False,
    filter_warning=True,
    sort=None,
    ignore_index=False,
):
    assert join == "outer"
    return cudf.concat(dfs, axis=axis, ignore_index=ignore_index)


try:

    from dask.dataframe.utils import group_split_dispatch, hash_object_dispatch
    from cudf.core.column import column, CategoricalColumn, StringColumn

    def _handle_string(s):
        if isinstance(s._column, StringColumn):
            s = s._hash()
        return s

    def safe_hash(df):
        frame = df.copy(deep=False)
        if isinstance(frame, cudf.DataFrame):
            for col in frame.columns:
                frame[col] = _handle_string(frame[col])
            return frame.hash_columns()
        else:
            return _handle_string(frame)

    @hash_object_dispatch.register((cudf.DataFrame, cudf.Series))
    def hash_object_cudf(frame, index=True):
        if index:
            return safe_hash(frame.reset_index())
        return safe_hash(frame)

    @hash_object_dispatch.register(cudf.Index)
    def hash_object_cudf_index(ind, index=None):

        if isinstance(ind, cudf.MultiIndex):
            return safe_hash(ind.to_frame(index=False))

        col = column.as_column(ind)
        if isinstance(col, StringColumn):
            col = col.as_numerical_column("int32")
        elif isinstance(col, CategoricalColumn):
            col = col.as_numerical
        return cudf.Series(col).hash_values()

    @group_split_dispatch.register(cudf.DataFrame)
    def group_split_cudf(df, c, k, ignore_index=False):
        return dict(
            zip(
                range(k),
                df.scatter_by_map(c, map_size=k, keep_index=not ignore_index),
            )
        )


except ImportError:
    pass
