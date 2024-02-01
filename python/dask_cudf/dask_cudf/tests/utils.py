# Copyright (c) 2022-2024, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import dask.dataframe as dd

import cudf

from dask_cudf.expr import DASK_EXPR_ENABLED


def _make_random_frame(nelem, npartitions=2, include_na=False):
    df = pd.DataFrame(
        {"x": np.random.random(size=nelem), "y": np.random.random(size=nelem)}
    )

    if include_na:
        df["x"][::2] = pd.NA

    gdf = cudf.DataFrame.from_pandas(df)
    dgf = dd.from_pandas(gdf, npartitions=npartitions)
    return df, dgf


_default_reason = "Not compatible with dask-expr"


def skip_module_dask_expr(reason=_default_reason):
    if DASK_EXPR_ENABLED:
        pytest.skip(
            allow_module_level=True,
            reason=reason,
        )


def skip_dask_expr(reason=_default_reason):
    return pytest.mark.skipif(DASK_EXPR_ENABLED, reason=reason)


def xfail_dask_expr(reason=_default_reason):
    return pytest.mark.xfail(DASK_EXPR_ENABLED, reason=reason)
