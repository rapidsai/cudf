# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_groupby_results_equal


@pytest.mark.parametrize("n", [0, 2, 10])
@pytest.mark.parametrize("by", ["a", ["a", "b"], ["a", "c"]])
def test_groupby_nth(n, by):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 3],
            "b": [1, 2, 2, 2, 1],
            "c": [1, 2, None, 4, 5],
            "d": ["a", "b", "c", "d", "e"],
        }
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby(by).nth(n)
    got = gdf.groupby(by).nth(n)

    assert_groupby_results_equal(expect, got, check_dtype=False)


def test_groupby_consecutive_operations():
    df = cudf.DataFrame([[1, np.nan], [1, 4], [5, 6]], columns=["A", "B"])
    pdf = df.to_pandas()

    gg = df.groupby("A")
    pg = pdf.groupby("A")

    actual = gg.nth(-1)
    expected = pg.nth(-1)

    assert_groupby_results_equal(actual, expected, check_dtype=False)

    actual = gg.nth(0)
    expected = pg.nth(0)

    assert_groupby_results_equal(actual, expected, check_dtype=False)

    actual = gg.cumsum()
    expected = pg.cumsum()

    assert_groupby_results_equal(actual, expected, check_dtype=False)

    actual = gg.cumcount()
    expected = pg.cumcount()

    assert_groupby_results_equal(actual, expected, check_dtype=False)

    actual = gg.cumsum()
    expected = pg.cumsum()

    assert_groupby_results_equal(actual, expected, check_dtype=False)
