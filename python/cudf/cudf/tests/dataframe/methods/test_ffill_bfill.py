# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import expect_warning_if


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"A": [1, 2, 3, np.nan, None, 6]}),
        pd.Series([1, 2, 3, None, np.nan, 5, 6, np.nan]),
    ],
)
@pytest.mark.parametrize("alias", ["bfill", "backfill"])
def test_dataframe_bfill(df, alias):
    gdf = cudf.from_pandas(df)

    with expect_warning_if(alias == "backfill"):
        actual = getattr(df, alias)()
    with expect_warning_if(alias == "backfill"):
        expected = getattr(gdf, alias)()
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"A": [1, 2, 3, np.nan, None, 6]}),
        pd.Series([1, 2, 3, None, np.nan, 5, 6, np.nan]),
    ],
)
@pytest.mark.parametrize("alias", ["ffill", "pad"])
def test_dataframe_ffill(df, alias):
    gdf = cudf.from_pandas(df)

    with expect_warning_if(alias == "pad"):
        actual = getattr(df, alias)()
    with expect_warning_if(alias == "pad"):
        expected = getattr(gdf, alias)()
    assert_eq(expected, actual)
