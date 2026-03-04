# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq, assert_groupby_results_equal


@pytest.mark.parametrize("method", ["average", "min", "max", "first", "dense"])
def test_rank_return_type_compatible_mode(method):
    # in compatible mode, rank() always returns floats
    pdf = pd.DataFrame({"a": [1, 1, 1, 2, 2], "b": [1, 2, 3, 4, 5]})
    with cudf.option_context("mode.pandas_compatible", True):
        df = cudf.from_pandas(pdf)
        result = df.groupby("a").rank(method=method)
    expect = pdf.groupby("a").rank(method=method)
    assert_eq(expect, result)
    assert result["b"].dtype == "float64"


@pytest.mark.parametrize("method", ["average", "min", "max", "first", "dense"])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_option", ["keep", "top", "bottom"])
@pytest.mark.parametrize("pct", [False, True])
def test_groupby_2keys_rank(method, ascending, na_option, pct):
    nelem = 20
    pdf = pd.DataFrame(
        {
            "x": np.arange(nelem),
            "y": np.arange(nelem),
            "z": np.concatenate([np.arange(nelem - 10), np.full(10, np.nan)]),
        }
    )
    gdf = cudf.from_pandas(pdf)
    expect_df = pdf.groupby(["x", "y"], sort=True).rank(
        method=method, ascending=ascending, na_option=na_option, pct=pct
    )
    got_df = gdf.groupby(["x", "y"], sort=True).rank(
        method=method, ascending=ascending, na_option=na_option, pct=pct
    )

    assert_groupby_results_equal(got_df, expect_df, check_dtype=False)


def test_groupby_rank_fails():
    gdf = cudf.DataFrame(
        {"x": [1, 2, 3, 4], "y": [1, 2, 3, 4], "z": [1, 2, 3, 4]}
    )
    with pytest.raises(NotImplementedError):
        gdf.groupby(["x", "y"]).rank(method="min", axis=1)
    gdf = cudf.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [[1, 2], [3, None, 5], None, [], [7, 8], [9]],
        }
    )
    with pytest.raises(NotImplementedError):
        gdf.groupby(["a"]).rank(method="min", axis=1)
