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


def test_reset_index_multiindex():
    df = cudf.DataFrame()
    df["id_1"] = ["a", "a", "b"]
    df["id_2"] = [0, 0, 1]
    df["val"] = [1, 2, 3]

    df_lookup = cudf.DataFrame()
    df_lookup["id_1"] = ["a", "b"]
    df_lookup["metadata"] = [0, 1]

    gddf = dask_cudf.from_cudf(df, npartitions=2)
    gddf_lookup = dask_cudf.from_cudf(df_lookup, npartitions=2)

    ddf = dd.from_pandas(df.to_pandas(), npartitions=2)
    ddf_lookup = dd.from_pandas(df_lookup.to_pandas(), npartitions=2)

    # Note: 'id_2' has wrong type (object) until after compute
    dd.assert_eq(
        gddf.groupby(by=["id_1", "id_2"])
        .val.sum()
        .reset_index()
        .merge(gddf_lookup, on="id_1")
        .compute(),
        ddf.groupby(by=["id_1", "id_2"])
        .val.sum()
        .reset_index()
        .merge(ddf_lookup, on="id_1"),
    )


@pytest.mark.parametrize("split_out", [1, 2, 3])
@pytest.mark.parametrize(
    "column", ["c", "d", "e", ["b", "c"], ["b", "d"], ["b", "e"]]
)
def test_groupby_split_out(split_out, column):
    df = pd.DataFrame(
        {
            "a": np.arange(8),
            "b": [1, 0, 0, 2, 1, 1, 2, 0],
            "c": [0, 1] * 4,
            "d": ["dog", "cat", "cat", "dog", "dog", "dog", "cat", "bird"],
        }
    )
    df["e"] = df["d"].astype("category")
    gdf = cudf.from_pandas(df)

    ddf = dd.from_pandas(df, npartitions=3)
    gddf = dask_cudf.from_cudf(gdf, npartitions=3)

    ddf_result = (
        ddf.groupby(column)
        .a.mean(split_out=split_out)
        .compute()
        .sort_values()
        .dropna()
    )
    gddf_result = (
        gddf.groupby(column)
        .a.mean(split_out=split_out)
        .compute()
        .sort_values()
    )

    dd.assert_eq(gddf_result, ddf_result, check_index=False)
