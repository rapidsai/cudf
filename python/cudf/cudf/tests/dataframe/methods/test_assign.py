# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_assign():
    gdf = cudf.DataFrame({"x": [1, 2, 3]})
    gdf2 = gdf.assign(y=gdf.x + 1)
    assert list(gdf.columns) == ["x"]
    assert list(gdf2.columns) == ["x", "y"]

    np.testing.assert_equal(gdf2.y.to_numpy(), [2, 3, 4])


@pytest.mark.parametrize(
    "mapping",
    [
        {"y": 1, "z": lambda df: df["x"] + df["y"]},
        {
            "x": lambda df: df["x"] * 2,
            "y": lambda df: 2,
            "z": lambda df: df["x"] / df["y"],
        },
    ],
)
def test_assign_callable(mapping):
    df = pd.DataFrame({"x": [1, 2, 3]})
    cdf = cudf.from_pandas(df)
    expect = df.assign(**mapping)
    actual = cdf.assign(**mapping)
    assert_eq(expect, actual)
