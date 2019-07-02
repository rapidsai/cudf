import numpy as np
import pandas as pd
import pytest

import dask.dataframe as dd

import cudf

import dask_cudf


@pytest.mark.parametrize("agg", ["sum", "mean", "count", "min", "max"])
def test_groupby_basic_aggs(agg):
    pdf = pd.DataFrame(
        {
            "x": np.random.randint(0, 5, size=10000),
            "y": np.random.normal(size=10000),
        }
    )

    gdf = cudf.DataFrame.from_pandas(pdf)

    ddf = dask_cudf.from_cudf(gdf, npartitions=5)

    a = getattr(gdf.groupby("x"), agg)().to_pandas()
    b = getattr(ddf.groupby("x"), agg)().compute().to_pandas()

    a.index.name = None
    a.name = None
    b.index.name = None
    b.name = None

    if agg == "count":
        a["y"] = a["y"].astype(np.int64)

    dd.assert_eq(a, b)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.groupby("x").agg({"y": "max"}),
        pytest.param(
            lambda df: df.groupby("x").y.agg(["sum", "max"]),
            marks=pytest.mark.skip,
        ),
    ],
)
def test_groupby_agg(func):
    pdf = pd.DataFrame(
        {
            "x": np.random.randint(0, 5, size=10000),
            "y": np.random.normal(size=10000),
        }
    )

    gdf = cudf.DataFrame.from_pandas(pdf)

    ddf = dask_cudf.from_cudf(gdf, npartitions=5)

    a = func(gdf).to_pandas()
    b = func(ddf).compute().to_pandas()

    a.index.name = None
    a.name = None
    b.index.name = None
    b.name = None

    dd.assert_eq(a, b)


@pytest.mark.xfail(reason="cudf issues")
@pytest.mark.parametrize(
    "func",
    [lambda df: df.groupby("x").std(), lambda df: df.groupby("x").y.std()],
)
def test_groupby_std(func):
    pdf = pd.DataFrame(
        {
            "x": np.random.randint(0, 5, size=10000),
            "y": np.random.normal(size=10000),
        }
    )

    gdf = cudf.DataFrame.from_pandas(pdf)

    ddf = dask_cudf.from_cudf(gdf, npartitions=5)

    a = func(gdf.to_pandas())
    b = func(ddf).compute().to_pandas()

    a.index.name = None
    a.name = None
    b.index.name = None

    dd.assert_eq(a, b)


# reason gotattr in cudf
@pytest.mark.parametrize(
    "func",
    [
        pytest.param(
            lambda df: df.groupby(["a", "b"]).x.sum(), marks=pytest.mark.xfail
        ),
        pytest.param(
            lambda df: df.groupby(["a", "b"]).sum(), marks=pytest.mark.xfail
        ),
        pytest.param(
            lambda df: df.groupby(["a", "b"]).agg({"x", "sum"}),
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_groupby_multi_column(func):
    pdf = pd.DataFrame(
        {
            "a": np.random.randint(0, 20, size=1000),
            "b": np.random.randint(0, 5, size=1000),
            "x": np.random.normal(size=1000),
        }
    )

    gdf = cudf.DataFrame.from_pandas(pdf)

    ddf = dask_cudf.from_cudf(gdf, npartitions=5)

    a = func(gdf).to_pandas()
    b = func(ddf).compute().to_pandas()

    dd.assert_eq(a, b)
