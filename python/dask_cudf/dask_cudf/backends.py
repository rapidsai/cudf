# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import warnings
from collections.abc import Iterator
from functools import partial

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.types import is_scalar

import dask.dataframe as dd
from dask import config
from dask.array.dispatch import percentile_lookup
from dask.dataframe.backends import (
    DataFrameBackendEntrypoint,
    PandasBackendEntrypoint,
)
from dask.dataframe.core import get_parallel_type, meta_nonempty
from dask.dataframe.dispatch import (
    categorical_dtype_dispatch,
    concat_dispatch,
    from_pyarrow_table_dispatch,
    group_split_dispatch,
    grouper_dispatch,
    hash_object_dispatch,
    is_categorical_dtype_dispatch,
    make_meta_dispatch,
    pyarrow_schema_dispatch,
    to_pyarrow_table_dispatch,
    tolist_dispatch,
    union_categoricals_dispatch,
)
from dask.dataframe.utils import (
    UNKNOWN_CATEGORIES,
    _nonempty_scalar,
    _scalar_from_dtype,
    make_meta_obj,
)
from dask.sizeof import sizeof as sizeof_dispatch
from dask.utils import Dispatch, is_arraylike

import cudf
from cudf.api.types import is_string_dtype
from cudf.utils.nvtx_annotation import _dask_cudf_nvtx_annotate

from .core import DataFrame, Index, Series

get_parallel_type.register(cudf.DataFrame, lambda _: DataFrame)
get_parallel_type.register(cudf.Series, lambda _: Series)
get_parallel_type.register(cudf.BaseIndex, lambda _: Index)


@meta_nonempty.register(cudf.BaseIndex)
@_dask_cudf_nvtx_annotate
def _nonempty_index(idx):
    if isinstance(idx, cudf.core.index.RangeIndex):
        return cudf.core.index.RangeIndex(2, name=idx.name)
    elif isinstance(idx, cudf.core.index.DatetimeIndex):
        start = "1970-01-01"
        data = np.array([start, "1970-01-02"], dtype=idx.dtype)
        values = cudf.core.column.as_column(data)
        return cudf.core.index.DatetimeIndex(values, name=idx.name)
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
    elif isinstance(idx, cudf.core.multiindex.MultiIndex):
        levels = [meta_nonempty(lev) for lev in idx.levels]
        codes = [[0, 0] for i in idx.levels]
        return cudf.core.multiindex.MultiIndex(
            levels=levels, codes=codes, names=idx.names
        )
    elif isinstance(idx._column, cudf.core.column.StringColumn):
        return cudf.Index(["cat", "dog"], name=idx.name)
    elif isinstance(idx, cudf.core.index.Index):
        return cudf.core.index.Index(
            np.arange(2, dtype=idx.dtype), name=idx.name
        )

    raise TypeError(f"Don't know how to handle index of type {type(idx)}")


def _nest_list_data(data, leaf_type):
    """
    Helper for _get_non_empty_data which creates
    nested list data
    """
    data = [data]
    while isinstance(leaf_type, cudf.ListDtype):
        leaf_type = leaf_type.element_type
        data = [data]
    return data


@_dask_cudf_nvtx_annotate
def _get_non_empty_data(s):
    if isinstance(s, cudf.core.column.CategoricalColumn):
        categories = (
            s.categories if len(s.categories) else [UNKNOWN_CATEGORIES]
        )
        codes = cudf.core.column.as_column(
            0,
            dtype=cudf._lib.types.size_type_dtype,
            length=2,
        )
        ordered = s.ordered
        data = cudf.core.column.build_categorical_column(
            categories=categories, codes=codes, ordered=ordered
        )
    elif isinstance(s, cudf.core.column.ListColumn):
        leaf_type = s.dtype.leaf_type
        if is_string_dtype(leaf_type):
            data = ["cat", "dog"]
        else:
            data = np.array([0, 1], dtype=leaf_type).tolist()
        data = _nest_list_data(data, s.dtype) * 2
        data = cudf.core.column.as_column(data, dtype=s.dtype)
    elif isinstance(s, cudf.core.column.StructColumn):
        struct_dtype = s.dtype
        data = [{key: None for key in struct_dtype.fields.keys()}] * 2
        data = cudf.core.column.as_column(data, dtype=s.dtype)
    elif is_string_dtype(s.dtype):
        data = pa.array(["cat", "dog"])
    elif isinstance(s.dtype, pd.DatetimeTZDtype):
        from cudf.utils.dtypes import get_time_unit

        data = cudf.date_range("2001-01-01", periods=2, freq=get_time_unit(s))
        data = data.tz_localize(str(s.dtype.tz))._column
    else:
        if pd.api.types.is_numeric_dtype(s.dtype):
            data = cudf.core.column.as_column(
                cp.arange(start=0, stop=2, dtype=s.dtype)
            )
        else:
            data = cudf.core.column.as_column(
                cp.arange(start=0, stop=2, dtype="int64")
            ).astype(s.dtype)
    return data


@meta_nonempty.register(cudf.Series)
@_dask_cudf_nvtx_annotate
def _nonempty_series(s, idx=None):
    if idx is None:
        idx = _nonempty_index(s.index)
    data = _get_non_empty_data(s._column)

    return cudf.Series(data, name=s.name, index=idx)


@meta_nonempty.register(cudf.DataFrame)
@_dask_cudf_nvtx_annotate
def meta_nonempty_cudf(x):
    idx = meta_nonempty(x.index)
    columns_with_dtype = dict()
    res = cudf.DataFrame(index=idx)
    for col in x._data.names:
        dtype = str(x._data[col].dtype)
        if dtype in ("list", "struct", "category"):
            # 1. Not possible to hash and store list & struct types
            #    as they can contain different levels of nesting or
            #    fields.
            # 2. Not possible to has `category` types as
            #    they often contain an underlying types to them.
            res._data[col] = _get_non_empty_data(x._data[col])
        else:
            if dtype not in columns_with_dtype:
                columns_with_dtype[dtype] = cudf.core.column.as_column(
                    _get_non_empty_data(x._data[col])
                )
            res._data[col] = columns_with_dtype[dtype]

    return res


@make_meta_dispatch.register((cudf.Series, cudf.DataFrame))
@_dask_cudf_nvtx_annotate
def make_meta_cudf(x, index=None):
    return x.head(0)


@make_meta_dispatch.register(cudf.BaseIndex)
@_dask_cudf_nvtx_annotate
def make_meta_cudf_index(x, index=None):
    return x[:0]


@_dask_cudf_nvtx_annotate
def _empty_series(name, dtype, index=None):
    if isinstance(dtype, str) and dtype == "category":
        return cudf.Series(
            [UNKNOWN_CATEGORIES], dtype=dtype, name=name, index=index
        ).iloc[:0]
    return cudf.Series([], dtype=dtype, name=name, index=index)


@make_meta_obj.register(object)
@_dask_cudf_nvtx_annotate
def make_meta_object_cudf(x, index=None):
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
        index = make_meta_dispatch(index)

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
                f"Expected iterable of tuples of (name, dtype), got {x}"
            )
        return cudf.DataFrame(
            {c: _empty_series(c, d, index=index) for (c, d) in x},
            columns=[c for c, d in x],
            index=index,
        )
    elif not hasattr(x, "dtype") and x is not None:
        # could be a string, a dtype object, or a python type. Skip `None`,
        # because it is implicitly converted to `dtype('f8')`, which we don't
        # want here.
        try:
            dtype = np.dtype(x)
            return _scalar_from_dtype(dtype)
        except Exception:
            # Continue on to next check
            pass

    if is_scalar(x):
        return _nonempty_scalar(x)

    raise TypeError(f"Don't know how to create metadata from {x}")


@concat_dispatch.register((cudf.DataFrame, cudf.Series, cudf.BaseIndex))
@_dask_cudf_nvtx_annotate
def concat_cudf(
    dfs,
    axis=0,
    join="outer",
    uniform=False,
    filter_warning=True,
    sort=None,
    ignore_index=False,
    **kwargs,
):
    assert join == "outer"

    ignore_order = kwargs.get("ignore_order", False)
    if ignore_order:
        raise NotImplementedError(
            "ignore_order parameter is not yet supported in dask-cudf"
        )

    return cudf.concat(dfs, axis=axis, ignore_index=ignore_index)


@categorical_dtype_dispatch.register(
    (cudf.DataFrame, cudf.Series, cudf.BaseIndex)
)
@_dask_cudf_nvtx_annotate
def categorical_dtype_cudf(categories=None, ordered=False):
    return cudf.CategoricalDtype(categories=categories, ordered=ordered)


@tolist_dispatch.register((cudf.Series, cudf.BaseIndex))
@_dask_cudf_nvtx_annotate
def tolist_cudf(obj):
    return obj.to_pandas().tolist()


@is_categorical_dtype_dispatch.register(
    (cudf.Series, cudf.BaseIndex, cudf.CategoricalDtype, Series)
)
@_dask_cudf_nvtx_annotate
def is_categorical_dtype_cudf(obj):
    return cudf.api.types._is_categorical_dtype(obj)


@grouper_dispatch.register((cudf.Series, cudf.DataFrame))
def get_grouper_cudf(obj):
    return cudf.core.groupby.Grouper


@percentile_lookup.register((cudf.Series, cp.ndarray, cudf.BaseIndex))
@_dask_cudf_nvtx_annotate
def percentile_cudf(a, q, interpolation="linear"):
    # Cudf dispatch to the equivalent of `np.percentile`:
    # https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
    a = cudf.Series(a)
    # a is series.
    n = len(a)
    if not len(a):
        return None, n
    if isinstance(q, Iterator):
        q = list(q)

    if cudf.api.types._is_categorical_dtype(a.dtype):
        result = cp.percentile(a.cat.codes, q, interpolation=interpolation)

        return (
            pd.Categorical.from_codes(
                result, a.dtype.categories, a.dtype.ordered
            ),
            n,
        )
    if np.issubdtype(a.dtype, np.datetime64):
        result = a.quantile(
            [i / 100.0 for i in q], interpolation=interpolation
        )

        if q[0] == 0:
            # https://github.com/dask/dask/issues/6864
            result[0] = min(result[0], a.min())
        return result.to_pandas(), n
    if not np.issubdtype(a.dtype, np.number):
        interpolation = "nearest"
    return (
        a.quantile(
            [i / 100.0 for i in q], interpolation=interpolation
        ).to_pandas(),
        n,
    )


@pyarrow_schema_dispatch.register((cudf.DataFrame,))
def _get_pyarrow_schema_cudf(obj, preserve_index=None, **kwargs):
    if kwargs:
        warnings.warn(
            "Ignoring the following arguments to "
            f"`pyarrow_schema_dispatch`: {list(kwargs)}"
        )

    return _cudf_to_table(
        meta_nonempty(obj), preserve_index=preserve_index
    ).schema


@to_pyarrow_table_dispatch.register(cudf.DataFrame)
def _cudf_to_table(obj, preserve_index=None, **kwargs):
    if kwargs:
        warnings.warn(
            "Ignoring the following arguments to "
            f"`to_pyarrow_table_dispatch`: {list(kwargs)}"
        )
    return obj.to_arrow(preserve_index=preserve_index)


@from_pyarrow_table_dispatch.register(cudf.DataFrame)
def _table_to_cudf(obj, table, self_destruct=None, **kwargs):
    # cudf ignores self_destruct.
    kwargs.pop("self_destruct", None)
    if kwargs:
        warnings.warn(
            f"Ignoring the following arguments to "
            f"`from_pyarrow_table_dispatch`: {list(kwargs)}"
        )
    return obj.from_arrow(table)


@union_categoricals_dispatch.register((cudf.Series, cudf.BaseIndex))
@_dask_cudf_nvtx_annotate
def union_categoricals_cudf(
    to_union, sort_categories=False, ignore_order=False
):
    return cudf.api.types._union_categoricals(
        to_union, sort_categories=False, ignore_order=False
    )


@hash_object_dispatch.register((cudf.DataFrame, cudf.Series))
@_dask_cudf_nvtx_annotate
def hash_object_cudf(frame, index=True):
    if index:
        frame = frame.reset_index()
    return frame.hash_values()


@hash_object_dispatch.register(cudf.BaseIndex)
@_dask_cudf_nvtx_annotate
def hash_object_cudf_index(ind, index=None):
    if isinstance(ind, cudf.MultiIndex):
        return ind.to_frame(index=False).hash_values()

    col = cudf.core.column.as_column(ind)
    return cudf.Series(col).hash_values()


@group_split_dispatch.register((cudf.Series, cudf.DataFrame))
@_dask_cudf_nvtx_annotate
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


@sizeof_dispatch.register(cudf.DataFrame)
@_dask_cudf_nvtx_annotate
def sizeof_cudf_dataframe(df):
    return int(
        sum(col.memory_usage for col in df._data.columns)
        + df._index.memory_usage()
    )


@sizeof_dispatch.register((cudf.Series, cudf.BaseIndex))
@_dask_cudf_nvtx_annotate
def sizeof_cudf_series_index(obj):
    return obj.memory_usage()


# TODO: Remove try/except when cudf is pinned to dask>=2023.10.0
try:
    from dask.dataframe.dispatch import partd_encode_dispatch

    @partd_encode_dispatch.register(cudf.DataFrame)
    def _simple_cudf_encode(_):
        # Basic pickle-based encoding for a partd k-v store
        import pickle

        import partd

        def join(dfs):
            if not dfs:
                return cudf.DataFrame()
            else:
                return cudf.concat(dfs)

        dumps = partial(pickle.dumps, protocol=pickle.HIGHEST_PROTOCOL)
        return partial(partd.Encode, dumps, pickle.loads, join)

except ImportError:
    pass


def _default_backend(func, *args, **kwargs):
    # Utility to call a dask.dataframe function with
    # the default ("pandas") backend

    # NOTE: Some `CudfBackendEntrypoint` methods need to
    # invoke the "pandas"-version of the same method, but
    # with custom kwargs (e.g. `engine`). In these cases,
    # an explicit "pandas" config context is needed to
    # avoid a recursive loop
    with config.set({"dataframe.backend": "pandas"}):
        return func(*args, **kwargs)


def _unsupported_kwargs(old, new, kwargs):
    # Utility to raise a meaningful error when
    # unsupported kwargs are encountered within
    # ``to_backend_dispatch``
    if kwargs:
        raise ValueError(
            f"Unsupported key-word arguments used in `to_backend` "
            f"for {old}-to-{new} conversion: {kwargs}"
        )


# Register cudf->pandas
to_pandas_dispatch = PandasBackendEntrypoint.to_backend_dispatch()


@to_pandas_dispatch.register((cudf.DataFrame, cudf.Series, cudf.Index))
def to_pandas_dispatch_from_cudf(data, nullable=False, **kwargs):
    _unsupported_kwargs("cudf", "pandas", kwargs)
    return data.to_pandas(nullable=nullable)


# Register pandas->cudf
to_cudf_dispatch = Dispatch("to_cudf_dispatch")


@to_cudf_dispatch.register((pd.DataFrame, pd.Series, pd.Index))
def to_cudf_dispatch_from_pandas(data, nan_as_null=None, **kwargs):
    _unsupported_kwargs("pandas", "cudf", kwargs)
    return cudf.from_pandas(data, nan_as_null=nan_as_null)


# Define "cudf" backend engine to be registered with Dask
class CudfBackendEntrypoint(DataFrameBackendEntrypoint):
    """Backend-entrypoint class for Dask-DataFrame

    This class is registered under the name "cudf" for the
    ``dask.dataframe.backends`` entrypoint in ``setup.cfg``.
    Dask-DataFrame will use the methods defined in this class
    in place of ``dask.dataframe.<creation-method>`` when the
    "dataframe.backend" configuration is set to "cudf":

    Examples
    --------
    >>> import dask
    >>> import dask.dataframe as dd
    >>> with dask.config.set({"dataframe.backend": "cudf"}):
    ...     ddf = dd.from_dict({"a": range(10)})
    >>> type(ddf)
    <class 'dask_cudf.core.DataFrame'>
    """

    @classmethod
    def to_backend_dispatch(cls):
        return to_cudf_dispatch

    @classmethod
    def to_backend(cls, data: dd.core._Frame, **kwargs):
        if isinstance(data._meta, (cudf.DataFrame, cudf.Series, cudf.Index)):
            # Already a cudf-backed collection
            _unsupported_kwargs("cudf", "cudf", kwargs)
            return data
        return data.map_partitions(cls.to_backend_dispatch(), **kwargs)

    @staticmethod
    def from_dict(
        data,
        npartitions,
        orient="columns",
        dtype=None,
        columns=None,
        constructor=cudf.DataFrame,
    ):
        return _default_backend(
            dd.from_dict,
            data,
            npartitions=npartitions,
            orient=orient,
            dtype=dtype,
            columns=columns,
            constructor=constructor,
        )

    @staticmethod
    def read_parquet(*args, engine=None, **kwargs):
        from dask_cudf.io.parquet import CudfEngine

        return _default_backend(
            dd.read_parquet,
            *args,
            engine=CudfEngine,
            **kwargs,
        )

    @staticmethod
    def read_json(*args, **kwargs):
        from dask_cudf.io.json import read_json

        return read_json(*args, **kwargs)

    @staticmethod
    def read_orc(*args, **kwargs):
        from dask_cudf.io import read_orc

        return read_orc(*args, **kwargs)

    @staticmethod
    def read_csv(*args, **kwargs):
        from dask_cudf.io import read_csv

        return read_csv(*args, **kwargs)

    @staticmethod
    def read_hdf(*args, **kwargs):
        # HDF5 reader not yet implemented in cudf
        warnings.warn(
            "read_hdf is not yet implemented in cudf/dask_cudf. "
            "Moving to cudf from pandas. Expect poor performance!"
        )
        return _default_backend(dd.read_hdf, *args, **kwargs).to_backend(
            "cudf"
        )


# Define "cudf" backend entrypoint for dask-expr
class CudfDXBackendEntrypoint(DataFrameBackendEntrypoint):
    """Backend-entrypoint class for Dask-Expressions

    This class is registered under the name "cudf" for the
    ``dask-expr.dataframe.backends`` entrypoint in ``setup.cfg``.
    Dask-DataFrame will use the methods defined in this class
    in place of ``dask_expr.<creation-method>`` when the
    "dataframe.backend" configuration is set to "cudf":

    Examples
    --------
    >>> import dask
    >>> import dask_expr
    >>> with dask.config.set({"dataframe.backend": "cudf"}):
    ...     ddf = dx.from_dict({"a": range(10)})
    >>> type(ddf._meta)
    <class 'cudf.core.dataframe.DataFrame'>
    """

    @classmethod
    def to_backend_dispatch(cls):
        return CudfBackendEntrypoint.to_backend_dispatch()

    @classmethod
    def to_backend(cls, *args, **kwargs):
        return CudfBackendEntrypoint.to_backend(*args, **kwargs)

    @staticmethod
    def from_dict(
        data,
        npartitions,
        orient="columns",
        dtype=None,
        columns=None,
        constructor=cudf.DataFrame,
    ):
        import dask_expr as dx

        return _default_backend(
            dx.from_dict,
            data,
            npartitions=npartitions,
            orient=orient,
            dtype=dtype,
            columns=columns,
            constructor=constructor,
        )

    @staticmethod
    def read_json(*args, engine="auto", **kwargs):
        return _default_backend(
            dd.read_json,
            *args,
            engine=(
                partial(cudf.read_json, engine=engine)
                if isinstance(engine, str)
                else engine
            ),
            **kwargs,
        )

    @staticmethod
    def read_orc(*args, **kwargs):
        from dask_expr import from_legacy_dataframe

        from dask_cudf.io.orc import read_orc as legacy_read_orc

        ddf = legacy_read_orc(*args, **kwargs)
        return from_legacy_dataframe(ddf)


# Import/register cudf-specific classes for dask-expr
try:
    import dask_cudf.expr  # noqa: F401
except ImportError:
    pass
