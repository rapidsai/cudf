# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf


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
