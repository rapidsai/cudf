# Copyright (c) 2025, NVIDIA CORPORATION.


import numpy as np
import pandas as pd
import pytest

from cudf import DataFrame
from cudf.testing import assert_eq


@pytest.mark.parametrize("n", [10, 5])
@pytest.mark.parametrize("op", ["nsmallest", "nlargest"])
@pytest.mark.parametrize("columns", ["a", ["b", "a"]])
def test_dataframe_nlargest_nsmallest(n, op, columns):
    nelem = 10
    rng = np.random.default_rng(seed=0)
    aa = rng.random(nelem)
    bb = rng.random(nelem)

    df = DataFrame({"a": aa, "b": bb})
    pdf = df.to_pandas()
    assert_eq(getattr(df, op)(n, columns), getattr(pdf, op)(n, columns))


@pytest.mark.parametrize(
    "sliceobj", [slice(1, None), slice(None, -1), slice(1, -1)]
)
def test_dataframe_nlargest_sliced(sliceobj):
    nelem = 20
    n = 10
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame()
    df["a"] = rng.random(nelem)
    df["b"] = rng.random(nelem)

    expect = df[sliceobj].nlargest(n, "a")
    gdf = DataFrame.from_pandas(df)
    got = gdf[sliceobj].nlargest(n, "a")
    assert (got.to_pandas() == expect).all().all()


@pytest.mark.parametrize(
    "sliceobj", [slice(1, None), slice(None, -1), slice(1, -1)]
)
def test_dataframe_nsmallest_sliced(sliceobj):
    nelem = 20
    n = 10
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame()
    df["a"] = rng.random(nelem)
    df["b"] = rng.random(nelem)

    expect = df[sliceobj].nsmallest(n, "a")
    gdf = DataFrame.from_pandas(df)
    got = gdf[sliceobj].nsmallest(n, "a")
    assert (got.to_pandas() == expect).all().all()
