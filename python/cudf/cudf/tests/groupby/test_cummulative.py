# Copyright (c) 2025, NVIDIA CORPORATION.
import numpy as np
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


@pytest.mark.parametrize(
    "func", ["cummin", "cummax", "cumcount", "cumsum", "cumprod"]
)
def test_groupby_2keys_scan(func):
    nelem = 20
    pdf = pd.DataFrame(np.ones((nelem, 3)), columns=["x", "y", "val"])
    expect_df = pdf.groupby(["x", "y"], sort=True).agg(func)
    gdf = cudf.from_pandas(pdf)
    got_df = gdf.groupby(["x", "y"], sort=True).agg(func)
    # pd.groupby.cumcount returns a series.
    if isinstance(expect_df, pd.Series):
        expect_df = expect_df.to_frame("val")

    assert_groupby_results_equal(got_df, expect_df)

    expect_df = getattr(pdf.groupby(["x", "y"], sort=True), func)()
    got_df = getattr(gdf.groupby(["x", "y"], sort=True), func)()
    assert_groupby_results_equal(got_df, expect_df)

    expect_df = getattr(pdf.groupby(["x", "y"], sort=True)[["x"]], func)()
    got_df = getattr(gdf.groupby(["x", "y"], sort=True)[["x"]], func)()
    assert_groupby_results_equal(got_df, expect_df)

    expect_df = getattr(pdf.groupby(["x", "y"], sort=True)["y"], func)()
    got_df = getattr(gdf.groupby(["x", "y"], sort=True)["y"], func)()
    assert_groupby_results_equal(got_df, expect_df)
