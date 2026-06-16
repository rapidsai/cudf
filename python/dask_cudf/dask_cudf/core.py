# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import textwrap
import warnings
from importlib import import_module

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
        Create a :class:`dask.dataframe.DataFrame` from a :class:`cudf.DataFrame`.

        This function is a thin wrapper around
        :func:`dask.dataframe.from_pandas`, accepting the same
        arguments (described below) excepting that it operates on cuDF
        rather than pandas objects.\n
        """
) + (
    textwrap.dedent(dd.from_pandas.__doc__)
    .replace("from_array", "dask.dataframe.from_array")
    .replace("read_csv", "dask.dataframe.read_csv")
)


def _deprecated_api(old_api, new_api=None, rec=None):
    def inner_func(*args, **kwargs):
        if new_api:
            # Use alternative
            msg = f"{old_api} is now deprecated. "
            msg += rec or f"Please use {new_api} instead."
            warnings.warn(msg, FutureWarning)
            new_attr = new_api.split(".")
            module = import_module(".".join(new_attr[:-1]))
            return getattr(module, new_attr[-1])(*args, **kwargs)

        # No alternative - raise an error
        raise NotImplementedError(
            f"{old_api} is no longer supported. " + (rec or "")
        )

    return inner_func
