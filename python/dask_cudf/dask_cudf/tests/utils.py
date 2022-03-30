import numpy as np
import pandas as pd

import dask.dataframe as dd

import cudf


def _make_empty_frame(npartitions=2):
    df = pd.DataFrame({"x": [], "y": []})
    gdf = cudf.DataFrame.from_pandas(df)
    dgf = dd.from_pandas(gdf, npartitions=npartitions)
    return dgf


def _make_random_frame(nelem, npartitions=2, include_na=False):
    df = pd.DataFrame(
        {"x": np.random.random(size=nelem), "y": np.random.random(size=nelem)}
    )

    if include_na:
        df["x"][::2] = pd.NA

    gdf = cudf.DataFrame.from_pandas(df)
    dgf = dd.from_pandas(gdf, npartitions=npartitions)
    return df, dgf


def _make_random_frame_float(nelem, npartitions=2):
    df = pd.DataFrame(
        {
            "x": np.random.randint(0, 5, size=nelem),
            "y": np.random.normal(size=nelem) + 1,
        }
    )
    gdf = cudf.from_pandas(df)
    dgf = dd.from_pandas(gdf, npartitions=npartitions)
    return df, dgf
