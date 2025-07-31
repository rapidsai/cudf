# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [[[1, 2, 3], None, [4], [], [5, 6]], [1, 2, 3, 4, 5]],
)
@pytest.mark.parametrize(
    "p_index",
    [
        None,
        ["ia", "ib", "ic", "id", "ie"],
        pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b")]
        ),
    ],
)
def test_explode(data, ignore_index, p_index):
    pdf = pd.Series(data, index=p_index, name="someseries")
    gdf = cudf.from_pandas(pdf)

    expect = pdf.explode(ignore_index)
    got = gdf.explode(ignore_index)

    assert_eq(expect, got, check_dtype=False)
