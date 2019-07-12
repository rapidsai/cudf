import pandas as pd

import cudf
import dask.dataframe as dd
import dask_cudf


def test_get_dummies(data):
    df = pd.DataFrame({"A": ["a", "b", "c", "a", "z"], "C": [1, 2, 3, 4, 5]})
    df["B"] = df["A"].astype('category')
    df["A"] = df["A"].astype('category')
    ddf = dd.from_pandas(df, npartitions=10)
    dd.assert_eq(dd.get_dummies(ddf).compute(), pd.get_dummies(df))
    gdf = cudf.from_pandas(df)
    gddf = dask_cudf.from_cudf(gdf, npartitions=10)
    dd.assert_eq(dd.get_dummies(ddf).compute(), dd.get_dummies(gddf),
                 check_dtype=False)

    df = pd.DataFrame({"A": ["a", "b", "c", "a", "z"],
                       "C": pd.Series([1, 2, 3, 4, 5], dtype='category')})
    df["B"] = df["A"].astype('category')
    df["A"] = df["A"].astype('category')
    ddf = dd.from_pandas(df, npartitions=10)
    dd.assert_eq(dd.get_dummies(ddf).compute(), pd.get_dummies(df))
    gdf = cudf.from_pandas(df)
    gddf = dask_cudf.from_cudf(gdf, npartitions=10)
    dd.assert_eq(dd.get_dummies(ddf).compute(), dd.get_dummies(gddf),
                 check_dtype=False)

    df = pd.DataFrame({"C": pd.Series([1, 2, 3, 4, 5])})
    ddf = dd.from_pandas(df, npartitions=10)
    dd.assert_eq(dd.get_dummies(ddf).compute(), pd.get_dummies(df))
    gdf = cudf.from_pandas(df)
    gddf = dask_cudf.from_cudf(gdf, npartitions=10)
    dd.assert_eq(dd.get_dummies(ddf).compute(), dd.get_dummies(gddf),
                 check_dtype=False)

    df = pd.DataFrame({"C": pd.CategoricalIndex([1, 2, 3, 4, 5])})
    ddf = dd.from_pandas(df, npartitions=10)
    dd.assert_eq(dd.get_dummies(ddf).compute(), pd.get_dummies(df))
    gdf = cudf.from_pandas(df)
    gddf = dask_cudf.from_cudf(gdf, npartitions=10)
    dd.assert_eq(dd.get_dummies(ddf).compute(), dd.get_dummies(gddf),
                 check_dtype=False)

    df = pd.DataFrame({"C": [], "A": []})
    ddf = dd.from_pandas(df, npartitions=10)
    dd.assert_eq(dd.get_dummies(ddf).compute(), pd.get_dummies(df))
    gdf = cudf.from_pandas(df)
    gddf = dask_cudf.from_cudf(gdf, npartitions=10)
    dd.assert_eq(dd.get_dummies(ddf).compute(), dd.get_dummies(gddf),
                 check_dtype=False)
