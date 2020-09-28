# Copyright (c) 2020, NVIDIA CORPORATION.

import pandas as pd
import pytest
import cudf as gd

from cudf.tests.utils import assert_eq


@gd.api.extensions.register_dataframe_accessor("point")
@pd.api.extensions.register_dataframe_accessor("point")
class PointsAccessor:
    def __init__(self, obj):
        self._validate(obj)
        self._obj = obj

    @staticmethod
    def _validate(obj):
        cols = obj.columns
        if not all([vertex in cols for vertex in ["x", "y"]]):
            raise AttributeError("Must have vertices 'x', 'y'.")

    @property
    def bounding_box(self):
        xs, ys = self._obj["x"], self._obj["y"]
        min_x, min_y, max_x, max_y = xs.min(), ys.min(), xs.max(), ys.max()

        return (min_x, min_y, max_x, max_y)


@pytest.mark.parametrize(
    "gdf", [gd.datasets.randomdata(nrows=6, dtypes={"x": int, "y": int})]
)
def test_dataframe_accessor(gdf):
    pdf = gdf.to_pandas()

    assert_eq(gdf.point.bounding_box, pdf.point.bounding_box)


@pytest.mark.parametrize(
    "gdf1", [gd.datasets.randomdata(nrows=1, dtypes={"x": int, "y": int})]
)
@pytest.mark.parametrize(
    "gdf2", [gd.datasets.randomdata(nrows=1, dtypes={"x": int, "y": int})]
)
def test_dataframe_accessor_idendity(gdf1, gdf2):
    """Test for accessor idendities
    - An object should hold persistent reference to the same accessor
    - Different objects should hold difference instances of the accessor
    """

    assert gdf1.point is gdf1.point
    assert not (gdf1.point is gdf2.point)


@pd.api.extensions.register_index_accessor("odd")
@pd.api.extensions.register_series_accessor("odd")
@gd.api.extensions.register_index_accessor("odd")
@gd.api.extensions.register_series_accessor("odd")
class OddRowAccessor:
    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, i):
        return self._obj[2 * i - 1]


@pytest.mark.parametrize("gidx", [gd.Index(list(range(0, 50)))])
def test_index_accessor(gidx):
    pidx = gidx.to_pandas()

    for i in range(1, 10):
        assert_eq(gidx.odd[i], pidx.odd[i])


@pytest.mark.parametrize("gs", [gd.Series(list(range(1, 50)))])
def test_series_accessor(gs):
    ps = gs.to_pandas()

    for i in range(1, 10):
        assert_eq(gs.odd[i], ps.odd[i])


@pytest.mark.parametrize(
    "gdf", [gd.datasets.randomdata(nrows=6, dtypes={"x": int, "y": int})]
)
@pytest.mark.parametrize("gidx", [gd.Index(list(range(1, 50)))])
@pytest.mark.parametrize("gs", [gd.Series(list(range(1, 50)))])
def test_accessor_space_separate(gdf, gidx, gs):
    assert not id(gdf._accessors) == id(gidx._accessors)
    assert not id(gidx._accessors) == id(gs._accessors)
    assert not id(gdf._accessors) == id(gs._accessors)
