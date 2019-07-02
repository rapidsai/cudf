import numpy as np
import pandas as pd
import pytest

import dask
import dask.dataframe as dd

import cudf


@pytest.mark.parametrize("by", ["a", "b"])
@pytest.mark.parametrize("nelem", [10, 100, 1000])
@pytest.mark.parametrize("nparts", [1, 2, 5, 10])
def test_sort_values(nelem, nparts, by):
    df = cudf.DataFrame()
    df["a"] = np.ascontiguousarray(np.arange(nelem)[::-1])
    df["b"] = np.arange(100, nelem + 100)
    ddf = dd.from_pandas(df, npartitions=nparts)

    with dask.config.set(scheduler="single-threaded"):
        got = ddf.sort_values(by=by).compute().to_pandas()
    expect = df.sort_values(by=by).to_pandas().reset_index(drop=True)
    pd.util.testing.assert_frame_equal(got, expect)


def test_sort_values_binned():
    np.random.seed(43)
    nelem = 100
    nparts = 5
    by = "a"
    df = cudf.DataFrame()
    df["a"] = np.random.randint(1, 5, nelem)
    ddf = dd.from_pandas(df, npartitions=nparts)

    parts = ddf.sort_values_binned(by=by).to_delayed()
    part_uniques = []
    for i, p in enumerate(parts):
        part = dask.compute(p)[0]
        part_uniques.append(set(part.a.unique()))

    # Partitions do not have intersecting keys
    for i in range(len(part_uniques)):
        for j in range(i + 1, len(part_uniques)):
            assert not (
                part_uniques[i] & part_uniques[j]
            ), "should have empty intersection"


def test_sort_binned_meta():
    df = cudf.DataFrame([("a", [0, 1, 2, 3, 4]), ("b", [5, 6, 7, 7, 8])])
    ddf = dd.from_pandas(df, npartitions=2).persist()

    ddf.sort_values_binned(by="b")
