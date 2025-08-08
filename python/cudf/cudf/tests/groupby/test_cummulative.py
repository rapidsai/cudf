# Copyright (c) 2025, NVIDIA CORPORATION.


import pandas as pd
import pytest

import cudf
from cudf.tests.groupby.testing import assert_groupby_results_equal


@pytest.mark.parametrize("index", [None, [1, 2, 3, 4]])
def test_groupby_cumcount(index):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 3, 4],
            "b": ["bob", "bob", "alice", "cooper"],
            "c": [1, 2, 3, 4],
        },
        index=index,
    )
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").cumcount(),
        gdf.groupby("a").cumcount(),
        check_dtype=False,
    )

    assert_groupby_results_equal(
        pdf.groupby(["a", "b", "c"]).cumcount(),
        gdf.groupby(["a", "b", "c"]).cumcount(),
        check_dtype=False,
    )

    sr = pd.Series(range(len(pdf)), index=index)
    assert_groupby_results_equal(
        pdf.groupby(sr).cumcount(),
        gdf.groupby(sr).cumcount(),
        check_dtype=False,
    )
