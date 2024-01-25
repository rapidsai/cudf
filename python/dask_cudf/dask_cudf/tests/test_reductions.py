# Copyright (c) 2021-2024, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

from dask import dataframe as dd

import cudf

import dask_cudf as dgd


def _make_random_frame(nelem, npartitions=2):
    df = pd.DataFrame(
        {
            "x": np.random.randint(0, 5, size=nelem),
            "y": np.random.normal(size=nelem) + 1,
        }
    )
    gdf = cudf.DataFrame.from_pandas(df)
    dgf = dgd.from_cudf(gdf, npartitions=npartitions)
    return df, dgf


_reducers = ["sum", "count", "mean", "var", "std", "min", "max"]


def _get_reduce_fn(name):
    def wrapped(series):
        fn = getattr(series, name)
        return fn()

    return wrapped


@pytest.mark.parametrize("reducer", _reducers)
def test_series_reduce(reducer):
    reducer = _get_reduce_fn(reducer)
    np.random.seed(0)
    size = 10
    df, gdf = _make_random_frame(size)

    got = reducer(gdf.x)
    exp = reducer(df.x)
    dd.assert_eq(got, exp)


@pytest.mark.parametrize(
    "data",
    [
        cudf.datasets.randomdata(
            nrows=10000,
            dtypes={"a": "category", "b": int, "c": float, "d": int},
        ),
        cudf.datasets.randomdata(
            nrows=10000,
            dtypes={"a": "category", "b": int, "c": float, "d": str},
        ),
        cudf.datasets.randomdata(
            nrows=10000, dtypes={"a": bool, "b": int, "c": float, "d": str}
        ),
    ],
)
@pytest.mark.parametrize(
    "op", ["max", "min", "sum", "prod", "mean", "var", "std"]
)
def test_rowwise_reductions(data, op):
    gddf = dgd.from_cudf(data, npartitions=10)
    pddf = gddf.to_dask_dataframe()

    if op in ("var", "std"):
        expected = getattr(pddf, op)(axis=1, ddof=0)
        got = getattr(gddf, op)(axis=1, ddof=0)
    else:
        expected = getattr(pddf, op)(axis=1)
        got = getattr(pddf, op)(axis=1)

    dd.assert_eq(expected.compute(), got.compute(), check_exact=False)
