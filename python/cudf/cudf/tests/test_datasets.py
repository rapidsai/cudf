import numpy as np

import cudf as gd
from cudf.testing._utils import assert_eq


def test_dataset_timeseries():
    gdf1 = gd.datasets.timeseries(
        dtypes={"x": int, "y": float}, freq="120s", seed=1
    )
    gdf2 = gd.datasets.timeseries(
        dtypes={"x": int, "y": float}, freq="120s", seed=1
    )

    assert_eq(gdf1, gdf2)

    assert gdf1["x"].head().dtype == int
    assert gdf1["y"].head().dtype == float
    assert gdf1.index.name == "timestamp"

    gdf = gd.datasets.timeseries(
        "2000",
        "2010",
        freq="2H",
        dtypes={"value": float, "name": "category", "id": int},
        seed=1,
    )

    assert gdf["value"].head().dtype == float
    assert gdf["id"].head().dtype == int
    assert gdf["name"].head().dtype == "category"

    gdf = gd.datasets.randomdata()
    assert gdf["id"].head().dtype == int
    assert gdf["x"].head().dtype == float
    assert gdf["y"].head().dtype == float
    assert len(gdf) == 10

    gdf = gd.datasets.randomdata(
        nrows=20, dtypes={"id": int, "a": int, "b": float}
    )
    assert gdf["id"].head().dtype == int
    assert gdf["a"].head().dtype == int
    assert gdf["b"].head().dtype == float
    assert len(gdf) == 20


def test_make_bool():
    n = 10
    state = np.random.RandomState(12)
    arr = gd.datasets.make_bool(n, state)
    assert np.alltrue(np.isin(arr, [True, False]))
    assert arr.size == n
    assert arr.dtype == bool
