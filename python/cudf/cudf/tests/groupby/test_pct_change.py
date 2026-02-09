# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import pytest

import cudf
from cudf.api.extensions import no_default
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.testing import assert_eq
from cudf.testing._utils import (
    expect_warning_if,
)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize(
    "data, gkey",
    [
        (
            {
                "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
                "val1": [5, 4, 6, 4, 8, 7, 4, 5, 2],
                "val2": [4, 5, 6, 1, 2, 9, 8, 5, 1],
                "val3": [4, 5, 6, 1, 2, 9, 8, 5, 1],
            },
            ["id"],
        ),
        (
            {
                "id": [0, 0, 0, 0, 1, 1, 1],
                "a": [1, 3, 4, 2.0, -3.0, 9.0, 10.0],
                "b": [10.0, 23, -4.0, 2, -3.0, None, 19.0],
            },
            ["id", "a"],
        ),
        (
            {
                "id": ["a", "a", "b", "b", "c", "c"],
                "val1": [None, None, None, None, None, None],
            },
            ["id"],
        ),
    ],
)
@pytest.mark.parametrize("periods", [-2, 0, 5])
@pytest.mark.parametrize("fill_method", ["ffill", "bfill", no_default, None])
def test_groupby_pct_change(data, gkey, periods, fill_method):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    with expect_warning_if(fill_method not in (no_default, None)):
        actual = gdf.groupby(gkey).pct_change(
            periods=periods, fill_method=fill_method
        )
    with expect_warning_if(
        (
            fill_method not in (no_default, None)
            or (fill_method is not None and pdf.isna().any().any())
        )
    ):
        expected = pdf.groupby(gkey).pct_change(
            periods=periods, fill_method=fill_method
        )

    assert_eq(expected, actual)


@pytest.mark.parametrize("periods", [-5, 5])
def test_groupby_pct_change_multiindex_dataframe(periods):
    gdf = cudf.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [1, 1, 2, 3],
            "c": [2, 3, 4, 5],
            "d": [6, 8, 9, 1],
        }
    ).set_index(["a", "b"])

    actual = gdf.groupby(level=["a", "b"]).pct_change(periods)
    expected = gdf.to_pandas().groupby(level=["a", "b"]).pct_change(periods)

    assert_eq(expected, actual)


def test_groupby_pct_change_empty_columns():
    gdf = cudf.DataFrame(columns=["id", "val1", "val2"])
    pdf = gdf.to_pandas()

    actual = gdf.groupby("id").pct_change()
    expected = pdf.groupby("id").pct_change()

    assert_eq(expected, actual)
