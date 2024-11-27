# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import textwrap

import dask.dataframe as dd
from dask.tokenize import tokenize

import cudf
from cudf.utils.performance_tracking import _dask_cudf_performance_tracking

# This module provides backward compatibility for legacy import patterns.
if dd.DASK_EXPR_ENABLED:
    from dask_cudf._expr.collection import (
        DataFrame,
        Index,
        Series,
    )
else:
    from dask_cudf._legacy.core import DataFrame, Index, Series  # noqa: F401


concat = dd.concat


@_dask_cudf_performance_tracking
def from_cudf(data, npartitions=None, chunksize=None, sort=True, name=None):
    from dask_cudf import QUERY_PLANNING_ON

    if isinstance(getattr(data, "index", None), cudf.MultiIndex):
        raise NotImplementedError(
            "dask_cudf does not support MultiIndex Dataframes."
        )

    # Dask-expr doesn't support the `name` argument
    name = {}
    if not QUERY_PLANNING_ON:
        name = {
            "name": name
            or ("from_cudf-" + tokenize(data, npartitions or chunksize))
        }

    return dd.from_pandas(
        data,
        npartitions=npartitions,
        chunksize=chunksize,
        sort=sort,
        **name,
    )


from_cudf.__doc__ = (
    textwrap.dedent(
        """
        Create a :class:`.DataFrame` from a :class:`cudf.DataFrame`.

        This function is a thin wrapper around
        :func:`dask.dataframe.from_pandas`, accepting the same
        arguments (described below) excepting that it operates on cuDF
        rather than pandas objects.\n
        """
    )
    # TODO: `dd.from_pandas.__doc__` is empty when
    # `DASK_DATAFRAME__QUERY_PLANNING=True`
    # since dask-expr does not provide a docstring for from_pandas.
    + textwrap.dedent(dd.from_pandas.__doc__ or "")
)
