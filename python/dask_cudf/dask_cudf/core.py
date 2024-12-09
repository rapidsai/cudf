# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import textwrap

import dask.dataframe as dd

import cudf
from cudf.utils.performance_tracking import _dask_cudf_performance_tracking

# This module provides backward compatibility for legacy import patterns.
from dask_cudf._expr.collection import (
    DataFrame,  # noqa: F401
    Index,  # noqa: F401
    Series,  # noqa: F401
)

concat = dd.concat


@_dask_cudf_performance_tracking
def from_cudf(data, npartitions=None, chunksize=None, sort=True, name=None):
    if isinstance(getattr(data, "index", None), cudf.MultiIndex):
        raise NotImplementedError(
            "dask_cudf does not support MultiIndex Dataframes."
        )

    return dd.from_pandas(
        data,
        npartitions=npartitions,
        chunksize=chunksize,
        sort=sort,
    )


from_cudf.__doc__ = textwrap.dedent(
    """
        Create a :class:`.DataFrame` from a :class:`cudf.DataFrame`.

        This function is a thin wrapper around
        :func:`dask.dataframe.from_pandas`, accepting the same
        arguments (described below) excepting that it operates on cuDF
        rather than pandas objects.\n
        """
) + textwrap.dedent(dd.from_pandas.__doc__ or "")
