# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import hvplot.pandas  # noqa: F401, needs to monkey patch pandas with this.
import numpy as np
import pandas as pd
import pytest

nsamps = 1000


def assert_hvplot_equal(expect, got):
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


pytestmark = pytest.mark.assert_eq(fn=assert_hvplot_equal)


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


def test_hvplot_barplot(df):
    return get_plot_info(df.hvplot.bar(x="category", y="y"))


def test_hvplot_scatterplot(df):
    return get_plot_info(df.hvplot.scatter(x="x", y="y"))


def test_hvplot_lineplot(df):
    return get_plot_info(df.hvplot.line(x="x", y="y"))


def test_hvplot_heatmap(df):
    return get_plot_info(df.hvplot.heatmap(x="x", y="y", C="y"))


def test_hvplot_hexbin(df):
    return get_plot_info(df.hvplot.hexbin(x="x", y="y", C="y"))
