# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import io

import pandas as pd
import pytest
import seaborn as sns


def assert_plots_equal(expect, got):
    # Both expect and got are now bytes from savefig
    assert expect == got


def _save_figure(ax):
    """Save the figure to an in-memory buffer and return the bytes.

    This avoids pickling issues with matplotlib Axes objects that contain
    itertools objects (which will lose pickle support in Python 3.14).
    """
    buf = io.BytesIO()
    ax.get_figure().savefig(buf, format="png")
    buf.seek(0)
    return buf.getvalue()


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
    return _save_figure(ax)


def test_scatter(df):
    ax = sns.scatterplot(data=df, x="x", y="y", hue="hue")
    return _save_figure(ax)


def test_lineplot_with_sns_data():
    df = sns.load_dataset("flights")
    ax = sns.lineplot(data=df, x="month", y="passengers")
    return _save_figure(ax)
