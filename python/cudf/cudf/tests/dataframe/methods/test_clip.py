# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ([2, 7.4], [4, 7.9]),
        ([2, 7.4], None),
        (
            None,
            [4, 7.9],
        ),
    ],
)
def test_dataframe_clip(lower, upper, inplace):
    pdf = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [7.1, 7.24, 7.5, 7.8, 8.11]}
    )
    gdf = cudf.from_pandas(pdf)

    got = gdf.clip(lower=lower, upper=upper, inplace=inplace)
    expect = pdf.clip(lower=lower, upper=upper, axis=1)

    if inplace is True:
        assert_eq(expect, gdf)
    else:
        assert_eq(expect, got)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [("b", "d"), ("b", None), (None, "c"), (None, None)],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_dataframe_category_clip(lower, upper, inplace):
    data = ["a", "b", "c", "d", "e"]
    pdf = pd.DataFrame({"a": data})
    gdf = cudf.from_pandas(pdf)
    gdf["a"] = gdf["a"].astype("category")

    expect = pdf.clip(lower=lower, upper=upper)
    got = gdf.clip(lower=lower, upper=upper, inplace=inplace)

    if inplace is True:
        assert_eq(expect, gdf.astype("str"))
    else:
        assert_eq(expect, got.astype("str"))


@pytest.mark.parametrize(
    ("lower", "upper"),
    [([2, 7.4], [4, 7.9, "d"]), ([2, 7.4, "a"], [4, 7.9, "d"])],
)
def test_dataframe_exceptions_for_clip(lower, upper):
    gdf = cudf.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [7.1, 7.24, 7.5, 7.8, 8.11]}
    )

    with pytest.raises(ValueError):
        gdf.clip(lower=lower, upper=upper)
