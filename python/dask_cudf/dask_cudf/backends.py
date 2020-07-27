# Copyright (c) 2020, NVIDIA CORPORATION.
import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa

from dask.dataframe.categorical import categorical_dtype_dispatch
from dask.dataframe.core import get_parallel_type, make_meta, meta_nonempty
from dask.dataframe.methods import concat_dispatch, tolist_dispatch
from dask.dataframe.utils import (
    UNKNOWN_CATEGORIES,
    _nonempty_scalar,
    _scalar_from_dtype,
    is_arraylike,
    is_scalar,
)

import cudf
from cudf.utils.dtypes import is_string_dtype

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
        values = cudf.core.column.build_categorical_column(
            categories=categories, codes=codes, ordered=ordered
        )
        return cudf.core.index.CategoricalIndex(values, name=idx.name)
    elif isinstance(idx, cudf.core.index.GenericIndex):
        return cudf.core.index.GenericIndex(
            np.arange(2, dtype=idx.dtype), name=idx.name
        )
    elif isinstance(idx, cudf.core.MultiIndex):
        levels = [meta_nonempty(lev) for lev in idx.levels]
        codes = [[0, 0] for i in idx.levels]
        return cudf.core.MultiIndex(
            levels=levels, codes=codes, names=idx.names
        )

    raise TypeError(
        "Don't know how to handle index of type {0}".format(type(idx))
    )


def _get_non_empty_data(s):
    if isinstance(s._column, cudf.core.column.CategoricalColumn):
        categories = (
            s._column.categories
            if len(s._column.categories)
            else [UNKNOWN_CATEGORIES]
        )
        codes = cp.zeros(2, dtype="int32")
        ordered = s._column.ordered
        data = column.build_categorical_column(
            categories=categories, codes=codes, ordered=ordered
        )
    elif is_string_dtype(s.dtype):
        data = pa.array(["cat", "dog"])
    else:
        if pd.api.types.is_numeric_dtype(s.dtype):
            data = column.as_column(cp.arange(start=0, stop=2, dtype=s.dtype))
        else:
            data = column.as_column(
                cp.arange(start=0, stop=2, dtype="int64")
            ).astype(s.dtype)
    return data


@meta_nonempty.register(cudf.Series)
def _nonempty_series(s, idx=None):
    if idx is None:
        idx = _nonempty_index(s.index)
    data = _get_non_empty_data(s)

    return cudf.Series(data, name=s.name, index=idx)


@meta_nonempty.register(cudf.DataFrame)
def meta_nonempty_cudf(x):
    idx = meta_nonempty(x.index)
    columns_with_dtype = dict()
    res = cudf.DataFrame(index=idx)
    for col in x._data.names:
        dtype = str(x._data[col].dtype)
        if dtype not in columns_with_dtype:
            columns_with_dtype[dtype] = cudf.core.column.as_column(
                _get_non_empty_data(x[col])
            )
        res._data[col] = columns_with_dtype[dtype]
    return res


@make_meta.register((cudf.Series, cudf.DataFrame))
def make_meta_cudf(x, index=None):
    return x.head(0)


@make_meta.register(cudf.Index)
def make_meta_cudf_index(x, index=None):
    return x[:0]


def _empty_series(name, dtype, index=None):
    if isinstance(dtype, str) and dtype == "category":
        return cudf.Series(
            [UNKNOWN_CATEGORIES], dtype=dtype, name=name, index=index
        ).iloc[:0]
    return cudf.Series([], dtype=dtype, name=name, index=index)


@make_meta.register(object)
def make_meta_object(x, index=None):
    """Create an empty cudf object containing the desired metadata.

    Parameters
    ----------
    x : dict, tuple, list, cudf.Series, cudf.DataFrame, cudf.Index,
        dtype, scalar
        To create a DataFrame, provide a `dict` mapping of `{name: dtype}`, or
        an iterable of `(name, dtype)` tuples. To create a `Series`, provide a
        tuple of `(name, dtype)`. If a cudf object, names, dtypes, and index
        should match the desired output. If a dtype or scalar, a scalar of the
        same dtype is returned.
    index :  cudf.Index, optional
        Any cudf index to use in the metadata. If none provided, a
        `RangeIndex` will be used.

    Examples
    --------
    >>> make_meta([('a', 'i8'), ('b', 'O')])
    Empty DataFrame
    Columns: [a, b]
    Index: []
    >>> make_meta(('a', 'f8'))
    Series([], Name: a, dtype: float64)
    >>> make_meta('i8')
    1
    """
    if hasattr(x, "_meta"):
        return x._meta
    elif is_arraylike(x) and x.shape:
        return x[:0]

    if index is not None:
        index = make_meta(index)

    if isinstance(x, dict):
        return cudf.DataFrame(
            {c: _empty_series(c, d, index=index) for (c, d) in x.items()},
            index=index,
        )
    if isinstance(x, tuple) and len(x) == 2:
        return _empty_series(x[0], x[1], index=index)
    elif isinstance(x, (list, tuple)):
        if not all(isinstance(i, tuple) and len(i) == 2 for i in x):
            raise ValueError(
                "Expected iterable of tuples of (name, dtype), got {0}".format(
                    x
                )
            )
        return cudf.DataFrame(
            {c: _empty_series(c, d, index=index) for (c, d) in x},
            columns=[c for c, d in x],
            index=index,
        )
    elif not hasattr(x, "dtype") and x is not None:
        # could be a string, a dtype object, or a python type. Skip `None`,
        # because it is implictly converted to `dtype('f8')`, which we don't
        # want here.
        try:
            dtype = np.dtype(x)
            return _scalar_from_dtype(dtype)
        except Exception:
            # Continue on to next check
            pass

    if is_scalar(x):
        return _nonempty_scalar(x)

    raise TypeError("Don't know how to create metadata from {0}".format(x))


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


@categorical_dtype_dispatch.register((cudf.DataFrame, cudf.Series, cudf.Index))
def categorical_dtype_cudf(categories=None, ordered=None):
    return cudf.CategoricalDtype(categories=categories, ordered=ordered)


@tolist_dispatch.register((cudf.Series, cudf.Index))
def tolist_cudf(obj):
    return obj.to_arrow().to_pylist()


try:

    from dask.dataframe.utils import group_split_dispatch, hash_object_dispatch

    from cudf.core.column import column

    def safe_hash(frame):
        index = frame.index
        if isinstance(frame, cudf.DataFrame):
            return cudf.Series(frame.hash_columns(), index=index)
        else:
            return cudf.Series(frame.hash_values(), index=index)

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
        return safe_hash(cudf.Series(col))

    @group_split_dispatch.register((cudf.Series, cudf.DataFrame))
    def group_split_cudf(df, c, k, ignore_index=False):
        return dict(
            zip(
                range(k),
                df.scatter_by_map(
                    c.astype(np.int32, copy=False),
                    map_size=k,
                    keep_index=not ignore_index,
                ),
            )
        )


except ImportError:
    pass
