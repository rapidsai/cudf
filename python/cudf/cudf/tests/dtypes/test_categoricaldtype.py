# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data", [None, [], ["a"], [1], [1.0], ["a", "b", "c"]]
)
def test_cdt_eq(data, categorical_ordered):
    dt = cudf.CategoricalDtype(categories=data, ordered=categorical_ordered)
    assert dt == "category"
    assert dt == dt
    assert dt == cudf.CategoricalDtype(
        categories=None, ordered=categorical_ordered
    )
    assert dt == cudf.CategoricalDtype(
        categories=data, ordered=categorical_ordered
    )
    assert dt != cudf.CategoricalDtype(
        categories=data, ordered=not categorical_ordered
    )


@pytest.mark.parametrize(
    "data", [None, [], ["a"], [1], [1.0], ["a", "b", "c"]]
)
def test_cdf_to_pandas(data, categorical_ordered):
    assert (
        pd.CategoricalDtype(data, categorical_ordered)
        == cudf.CategoricalDtype(
            categories=data, ordered=categorical_ordered
        ).to_pandas()
    )


@pytest.mark.parametrize(
    "categories",
    [
        [],
        [1, 2, 3],
        pd.Series(["a", "c", "b"], dtype="category"),
        pd.Series([1, 2, 3, 4, -100], dtype="category"),
    ],
)
def test_categorical_dtype(categories, categorical_ordered):
    expected = pd.CategoricalDtype(
        categories=categories, ordered=categorical_ordered
    )
    got = cudf.CategoricalDtype(
        categories=categories, ordered=categorical_ordered
    )
    assert_eq(expected, got)

    expected = pd.CategoricalDtype(categories=categories)
    got = cudf.CategoricalDtype(categories=categories)
    assert_eq(expected, got)


def test_categorical_dtype_ordered_not_settable():
    with pytest.raises(AttributeError):
        cudf.CategoricalDtype().ordered = False
