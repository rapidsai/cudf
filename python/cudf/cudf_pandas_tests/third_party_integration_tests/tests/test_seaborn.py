# Copyright (c) 2023-2024, NVIDIA CORPORATION.
import pandas as pd
import pytest
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pandas._testing import assert_equal


def assert_plots_equal(expect, got):
    if isinstance(expect, Axes) and isinstance(got, Axes):
        for expect_ch, got_ch in zip(
            expect.get_children(), got.get_children()
        ):
            assert type(expect_ch) is type(got_ch)
            if isinstance(expect_ch, Line2D):
                assert_equal(expect_ch.get_xdata(), got_ch.get_xdata())
                assert_equal(expect_ch.get_ydata(), got_ch.get_ydata())
            elif isinstance(expect_ch, Rectangle):
                assert expect_ch.get_height() == got_ch.get_height()
    elif isinstance(expect, PathCollection) and isinstance(
        got, PathCollection
    ):
        assert_equal(expect.get_offsets()[:, 0], got.get_offsets()[:, 0])
        assert_equal(expect.get_offsets()[:, 1], got.get_offsets()[:, 1])
    else:
        assert_equal(expect, got)


pytestmark = pytest.mark.assert_eq(fn=assert_plots_equal)


@pytest.fixture(scope="module")
def df():
    df = pd.DataFrame(
        {
            "x": [2, 3, 4, 5, 11],
            "y": [4, 3, 2, 1, 15],
            "hue": ["c", "a", "b", "b", "a"],
        }
    )
    return df


def test_bar(df):
    ax = sns.barplot(data=df, x="x", y="y")
    return ax


def test_scatter(df):
    ax = sns.scatterplot(data=df, x="x", y="y", hue="hue")
    return ax


def test_lineplot_with_sns_data():
    df = sns.load_dataset("flights")
    ax = sns.lineplot(data=df, x="month", y="passengers")
    return ax
