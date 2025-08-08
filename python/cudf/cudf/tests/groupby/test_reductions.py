# Copyright (c) 2025, NVIDIA CORPORATION.


import numpy as np
import pandas as pd

import cudf
from cudf.tests.groupby.testing import assert_groupby_results_equal


def test_groupby_mean():
    pdf = pd.DataFrame(np.ones((20, 3)), columns=["x", "y", "val"])
    gdf = cudf.DataFrame(pdf)
    got_df = gdf.groupby(["x", "y"]).mean()
    expect_df = pdf.groupby(["x", "y"]).mean()
    assert_groupby_results_equal(got_df, expect_df)


def test_groupby_mean_3level():
    pdf = pd.DataFrame(np.ones((20, 4)), columns=["x", "y", "val", "z"])
    gdf = cudf.DataFrame(pdf)
    bys = list("xyz")
    got_df = pdf.groupby(bys).mean()
    expect_df = gdf.groupby(bys).mean()
    assert_groupby_results_equal(got_df, expect_df)
