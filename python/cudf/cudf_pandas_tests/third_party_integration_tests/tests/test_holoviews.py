# Copyright (c) 2023-2024, NVIDIA CORPORATION.
import holoviews as hv
import numpy as np
import pandas as pd
import pytest

nsamps = 1000
hv.extension("bokeh")  # load holoviews extension


def assert_holoviews_equal(expect, got):
    expect_data, expect_ndims, expect_kdims, expect_vdims, expect_shape = (
        expect
    )
    got_data, got_ndims, got_kdims, got_vdims, got_shape = got

    if isinstance(expect_data, dict):
        np.testing.assert_allclose(expect_data["x"], got_data["x"])
        np.testing.assert_allclose(
            expect_data["Frequency"], got_data["Frequency"]
        )
    else:
        pd._testing.assert_frame_equal(expect_data, got_data)
    assert expect_ndims == got_ndims
    assert expect_kdims == got_kdims
    assert expect_vdims == got_vdims
    assert expect_shape == got_shape


pytestmark = pytest.mark.assert_eq(fn=assert_holoviews_equal)


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "x": rng.random(nsamps),
            "y": rng.random(nsamps),
            "category": rng.integers(0, 10, nsamps),
            "category2": rng.integers(0, 10, nsamps),
        }
    )


def get_plot_info(plot):
    return (
        plot.data,
        plot.ndims,
        plot.kdims,
        plot.vdims,
        plot.shape,
    )


def test_holoviews_barplot(df):
    return get_plot_info(hv.Bars(df, kdims="category", vdims="y"))


def test_holoviews_scatterplot(df):
    return get_plot_info(hv.Scatter(df, kdims="x", vdims="y"))


def test_holoviews_curve(df):
    return get_plot_info(hv.Curve(df, kdims="category", vdims="y"))


def test_holoviews_heatmap(df):
    return get_plot_info(
        hv.HeatMap(df, kdims=["category", "category2"], vdims="y")
    )


@pytest.mark.skip(
    reason="AttributeError: 'ndarray' object has no attribute '_fsproxy_wrapped'"
)
def test_holoviews_histogram(df):
    return get_plot_info(hv.Histogram(df.values))


def test_holoviews_hexbin(df):
    return get_plot_info(hv.HexTiles(df, kdims=["x", "y"], vdims="y"))
