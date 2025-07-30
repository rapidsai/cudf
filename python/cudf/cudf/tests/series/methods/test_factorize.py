# Copyright (c) 2025, NVIDIA CORPORATION.

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 2, 1],
        [1, 2, None, 3, 1, 1],
        [],
        ["a", "b", "c", None, "z", "a"],
    ],
)
@pytest.mark.parametrize("use_na_sentinel", [True, False])
@pytest.mark.parametrize("sort", [True, False])
def test_series_factorize_use_na_sentinel(data, use_na_sentinel, sort):
    gsr = cudf.Series(data)
    psr = gsr.to_pandas(nullable=True)

    expected_labels, expected_cats = psr.factorize(
        use_na_sentinel=use_na_sentinel, sort=sort
    )
    actual_labels, actual_cats = gsr.factorize(
        use_na_sentinel=use_na_sentinel, sort=sort
    )
    assert_eq(expected_labels, actual_labels.get())
    assert_eq(expected_cats, actual_cats.to_pandas(nullable=True))
