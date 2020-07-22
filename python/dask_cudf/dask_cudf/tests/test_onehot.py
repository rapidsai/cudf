import pandas as pd
import pytest

from dask import dataframe as dd

import dask_cudf

import cudf


def test_get_dummies_cat():
    df = pd.DataFrame({"C": [], "A": []})
    ddf = dd.from_pandas(df, npartitions=10)
    dd.assert_eq(dd.get_dummies(ddf).compute(), pd.get_dummies(df))
    gdf = cudf.from_pandas(df)
    gddf = dask_cudf.from_cudf(gdf, npartitions=10)
    dd.assert_eq(
        dd.get_dummies(ddf).compute(),
        dd.get_dummies(gddf).compute(),
        check_dtype=False,
    )

    df = pd.DataFrame({"A": ["a", "b", "c", "a", "z"], "C": [1, 2, 3, 4, 5]})
    df["B"] = df["A"].astype("category")
    df["A"] = df["A"].astype("category")
    ddf = dd.from_pandas(df, npartitions=10)
    dd.assert_eq(dd.get_dummies(ddf).compute(), pd.get_dummies(df))
    gdf = cudf.from_pandas(df)
    gddf = dask_cudf.from_cudf(gdf, npartitions=10)
    dd.assert_eq(
        dd.get_dummies(ddf).compute(),
        dd.get_dummies(gddf).compute(),
        check_dtype=False,
    )

    df = pd.DataFrame(
        {
            "A": ["a", "b", "c", "a", "z"],
            "C": pd.Series([1, 2, 3, 4, 5], dtype="category"),
        }
    )
    df["B"] = df["A"].astype("category")
    df["A"] = df["A"].astype("category")
    ddf = dd.from_pandas(df, npartitions=10)
    dd.assert_eq(dd.get_dummies(ddf).compute(), pd.get_dummies(df))
    gdf = cudf.from_pandas(df)
    gddf = dask_cudf.from_cudf(gdf, npartitions=10)
    dd.assert_eq(
        dd.get_dummies(ddf).compute(),
        dd.get_dummies(gddf).compute(),
        check_dtype=False,
    )


def test_get_dummies_non_cat():
    df = pd.DataFrame({"C": pd.Series([1, 2, 3, 4, 5])})
    ddf = dd.from_pandas(df, npartitions=10)
    dd.assert_eq(dd.get_dummies(ddf).compute(), pd.get_dummies(df))
    with pytest.raises(NotImplementedError):
        dd.get_dummies(ddf, columns=["C"]).compute()
    gdf = cudf.from_pandas(df)
    gddf = dask_cudf.from_cudf(gdf, npartitions=10)
    with pytest.raises(NotImplementedError):
        dd.get_dummies(gddf, columns=["C"]).compute()
    dd.assert_eq(
        dd.get_dummies(ddf).compute(),
        dd.get_dummies(gddf).compute(),
        check_dtype=False,
    )


def test_get_dummies_cat_index():
    df = pd.DataFrame({"C": pd.CategoricalIndex([1, 2, 3, 4, 5])})
    ddf = dd.from_pandas(df, npartitions=10)
    dd.assert_eq(dd.get_dummies(ddf).compute(), pd.get_dummies(df))
    gdf = cudf.from_pandas(df)
    gddf = dask_cudf.from_cudf(gdf, npartitions=10)
    dd.assert_eq(
        dd.get_dummies(ddf).compute(),
        dd.get_dummies(gddf).compute(),
        check_dtype=False,
    )


def test_get_dummies_large():
    gdf = cudf.datasets.randomdata(
        nrows=200000,
        dtypes={
            "C": int,
            "first": "category",
            "b": float,
            "second": "category",
        },
    )
    df = gdf.to_pandas()
    ddf = dd.from_pandas(df, npartitions=25)
    dd.assert_eq(dd.get_dummies(ddf).compute(), pd.get_dummies(df))
    gddf = dask_cudf.from_cudf(gdf, npartitions=25)
    dd.assert_eq(
        dd.get_dummies(ddf).compute(),
        dd.get_dummies(gddf).compute(),
        check_dtype=False,
    )
