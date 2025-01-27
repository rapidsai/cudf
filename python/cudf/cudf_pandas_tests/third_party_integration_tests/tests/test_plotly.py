# Copyright (c) 2023-2024, NVIDIA CORPORATION.
import numpy as np
import pandas as pd
import plotly.express as px
import pytest

nsamps = 100


def assert_plotly_equal(expect, got):
    assert type(expect) is type(got)
    if isinstance(expect, dict):
        assert expect.keys() == got.keys()
        for k in expect.keys():
            assert_plotly_equal(expect[k], got[k])
    elif isinstance(got, list):
        assert len(expect) == len(got)
        for i in range(len(expect)):
            assert_plotly_equal(expect[i], got[i])
    elif isinstance(expect, np.ndarray):
        np.testing.assert_allclose(expect, got)
    else:
        assert expect == got


pytestmark = pytest.mark.assert_eq(fn=assert_plotly_equal)


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


def test_plotly_scatterplot(df):
    return px.scatter(df, x="x", y="y").to_plotly_json()


def test_plotly_lineplot(df):
    return px.line(df, x="category", y="y").to_plotly_json()


def test_plotly_barplot(df):
    return px.bar(df, x="category", y="y").to_plotly_json()


def test_plotly_histogram(df):
    return px.histogram(df, x="category").to_plotly_json()


def test_plotly_pie(df):
    return px.pie(df, values="category", names="category2").to_plotly_json()


def test_plotly_heatmap(df):
    return px.density_heatmap(df, x="category", y="category2").to_plotly_json()


def test_plotly_boxplot(df):
    return px.box(df, x="category", y="y").to_plotly_json()
