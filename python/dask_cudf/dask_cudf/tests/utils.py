# Copyright (c) 2022-2024, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest
from packaging.version import Version

import dask
import dask.dataframe as dd

import cudf

from dask_cudf import QUERY_PLANNING_ON

if QUERY_PLANNING_ON:
    DASK_VERSION = Version(dask.__version__)
else:
    DASK_VERSION = None


def _make_random_frame(nelem, npartitions=2, include_na=False):
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {"x": rng.random(size=nelem), "y": rng.random(size=nelem)}
    )

    if include_na:
        df["x"][::2] = pd.NA

    gdf = cudf.DataFrame.from_pandas(df)
    dgf = dd.from_pandas(gdf, npartitions=npartitions)
    return df, dgf


_default_reason = "Not compatible with dask-expr"


def skip_dask_expr(reason=_default_reason, lt_version=None):
    if lt_version is not None:
        skip = QUERY_PLANNING_ON and DASK_VERSION < Version(lt_version)
    else:
        skip = QUERY_PLANNING_ON
    return pytest.mark.skipif(skip, reason=reason)


def xfail_dask_expr(reason=_default_reason, lt_version=None):
    if lt_version is not None:
        xfail = QUERY_PLANNING_ON and DASK_VERSION < Version(lt_version)
    else:
        xfail = QUERY_PLANNING_ON
    return pytest.mark.xfail(xfail, reason=reason)


def require_dask_expr(reason="requires dask-expr"):
    return pytest.mark.skipif(not QUERY_PLANNING_ON, reason=reason)
