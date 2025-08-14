# Copyright (c) 2025, NVIDIA CORPORATION.
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
