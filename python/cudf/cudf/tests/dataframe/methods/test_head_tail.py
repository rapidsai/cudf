# Copyright (c) 2025, NVIDIA CORPORATION.


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("nelem", [0, 10])
def test_head_tail(nelem, numeric_types_as_str):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "a": rng.integers(0, 1000, nelem).astype(numeric_types_as_str),
            "b": rng.integers(0, 1000, nelem).astype(numeric_types_as_str),
        }
    )
    gdf = cudf.from_pandas(pdf)

    assert_eq(gdf.head(), pdf.head())
    assert_eq(gdf.head(3), pdf.head(3))
    assert_eq(gdf.head(-2), pdf.head(-2))
    assert_eq(gdf.head(0), pdf.head(0))

    assert_eq(gdf["a"].head(), pdf["a"].head())
    assert_eq(gdf["a"].head(3), pdf["a"].head(3))
    assert_eq(gdf["a"].head(-2), pdf["a"].head(-2))

    assert_eq(gdf.tail(), pdf.tail())
    assert_eq(gdf.tail(3), pdf.tail(3))
    assert_eq(gdf.tail(-2), pdf.tail(-2))
    assert_eq(gdf.tail(0), pdf.tail(0))

    assert_eq(gdf["a"].tail(), pdf["a"].tail())
    assert_eq(gdf["a"].tail(3), pdf["a"].tail(3))
    assert_eq(gdf["a"].tail(-2), pdf["a"].tail(-2))


def test_tail_for_string():
    gdf = cudf.DataFrame({"id": ["a", "b"], "v": [1, 2]})
    assert_eq(gdf.tail(3), gdf.to_pandas().tail(3))
