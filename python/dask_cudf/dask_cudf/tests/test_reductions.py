import numpy as np
import pandas as pd
import pytest

from dask.dataframe.utils import assert_eq

import cudf as gd

import dask_cudf as dgd


def _make_random_frame(nelem, npartitions=2):
    df = pd.DataFrame(
        {
            "x": np.random.randint(0, 5, size=nelem),
            "y": np.random.normal(size=nelem) + 1,
        }
    )
    gdf = gd.DataFrame.from_pandas(df)
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
    assert_eq(got, exp)
