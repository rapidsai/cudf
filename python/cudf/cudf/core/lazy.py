# Copyright (c) 2024, NVIDIA CORPORATION.


from typing import Optional

import pandas
import pandas.core.groupby

import cudf
import cudf.core.groupby.groupby
import cudf.pandas
from cudf.options import get_option
from cudf.pandas.fast_slow_proxy import (
    _Unusable,
    make_final_proxy_type,
    make_intermediate_proxy_type,
)
from cudf.utils import docutils

_parse_lazy_argument_docstring = """
Determine whether to use lazy or regular dataframes.

The `arg` argument takes precedence thus if not None, the boolean value
of the `arg` argument is returned. A ValueError is raised if `arg` is
True and dask_cudf/dask-expr isn't available.

If `arg` is None, True is returned if the cudf `lazy` option is True
and dask_cudf/dask-expr is available. Otherwise, False is returned.

Parameters
----------
arg
    Optional boolean argument that takes precedence over the cudf `lazy`
    option.

Returns
-------
The boolean answer. If True, dask_cudf is guaranteed to be available.
"""

try:
    import dask_cudf
    import dask_cudf.expr._collection
    import dask_cudf.expr._groupby

except ImportError:

    @docutils.doc_apply(_parse_lazy_argument_docstring)
    def parse_lazy_argument(arg: Optional[bool]) -> bool:
        if arg is True:
            raise ValueError("Using the 'lazy' option requires dask_cudf")
        return False
else:
    # When cudf.pandas acceleration mode, we get the slow "fall back"
    # classes from the accelerated Pandas module.
    if cudf.pandas.LOADED:
        slow_DataFrame = pandas.DataFrame
        slow_Series = pandas.Series
        slow_DataFrameGroupBy = pandas.core.groupby.DataFrameGroupBy
        slow_SeriesGroupBy = pandas.core.groupby.SeriesGroupBy
    else:
        slow_DataFrame = cudf.DataFrame
        slow_Series = cudf.Series
        slow_DataFrameGroupBy = cudf.core.groupby.groupby.DataFrameGroupBy
        slow_SeriesGroupBy = cudf.core.groupby.groupby.SeriesGroupBy

    def _fast_to_slow(
        df: dask_cudf.expr._collection.DataFrame,
    ) -> cudf.DataFrame:
        return df.compute()

    def _slow_method(name: str):
        def f(x):
            return getattr(x._fsproxy_slow, name)()

        return f

    DataFrame = make_final_proxy_type(
        "DataFrame",
        dask_cudf.expr._collection.DataFrame,
        slow_DataFrame,
        fast_to_slow=_fast_to_slow,
        slow_to_fast=_Unusable(),
        additional_attributes={
            "__str__": _slow_method("__str__"),
            "__repr__": _slow_method("__repr__"),
            "to_pandas": _slow_method("to_pandas"),
        },
    )
    cudf.DataFrame.register(DataFrame)

    Series = make_final_proxy_type(
        "Series",
        dask_cudf.expr._collection.Series,
        slow_Series,
        fast_to_slow=_fast_to_slow,
        slow_to_fast=_Unusable(),
        additional_attributes={
            "__str__": _slow_method("__str__"),
            "__repr__": _slow_method("__repr__"),
            "to_pandas": _slow_method("to_pandas"),
        },
    )
    cudf.Series.register(Series)

    DataFrameGroupBy = make_intermediate_proxy_type(
        "DataFrameGroupBy",
        dask_cudf.expr._groupby.GroupBy,
        slow_DataFrameGroupBy,
    )

    SeriesGroupBy = make_intermediate_proxy_type(
        "SeriesGroupBy",
        dask_cudf.expr._groupby.SeriesGroupBy,
        slow_SeriesGroupBy,
    )

    @docutils.doc_apply(_parse_lazy_argument_docstring)
    def parse_lazy_argument(arg: Optional[bool]) -> bool:
        if arg is None:
            arg = get_option("lazy")
        return arg

    def lazy_wrap_dataframe(
        df: cudf.DataFrame | dask_cudf.DataFrame,
        *,
        noop_on_error: bool = False,
    ) -> cudf.DataFrame | dask_cudf.DataFrame:
        """Wrap a datafrane in a lazy proxy.

        Parameters
        ----------
        df
            The dataframe to wrap.
        noop_on_error
            If True, NotImplementedError and ValueError exceptions are
            ignored and `df` is returned as-is.

        Returns
        -------
        The wrapped dataframe or possible `df` as-is if noop_on_error is True.
        """
        try:
            if not isinstance(df, dask_cudf.DataFrame):
                df = dask_cudf.from_cudf(df, npartitions=1)
            return DataFrame._fsproxy_wrap(df, func=None)
        except (NotImplementedError, ValueError):
            if noop_on_error:
                return df
            raise

    def lazy_wrap_series(
        ser: cudf.Series | dask_cudf.Series, *, noop_on_error: bool = False
    ) -> cudf.Series | dask_cudf.Series:
        """Wrap a Series in a lazy proxy.

        Parameters
        ----------
        df
            The series to wrap.
        noop_on_error
            If True, NotImplementedError and ValueError exceptions are
            ignored and `df` is returned as-is.

        Returns
        -------
        The wrapped series or possible `ser` as-is if noop_on_error is True.
        """
        try:
            if not isinstance(ser, dask_cudf.Series):
                ser = dask_cudf.from_cudf(ser, npartitions=1)
            return Series._fsproxy_wrap(ser, func=None)
        except (NotImplementedError, ValueError):
            if noop_on_error:
                return ser
            raise
