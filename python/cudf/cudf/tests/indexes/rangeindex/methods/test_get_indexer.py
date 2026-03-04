# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "rng",
    [
        range(1, 20, 3),
        range(20, 35, 3),
        range(35, 77, 3),
        range(77, 110, 3),
    ],
)
@pytest.mark.parametrize("method", [None, "ffill", "bfill", "nearest"])
@pytest.mark.parametrize("tolerance", [None, 0, 1, 13, 20])
def test_get_indexer_rangeindex(rng, method, tolerance):
    key = list(rng)
    pi = pd.RangeIndex(3, 100, 4)
    gi = cudf.from_pandas(pi)

    expected = pi.get_indexer(
        key, method=method, tolerance=None if method is None else tolerance
    )
    got = gi.get_indexer(
        key, method=method, tolerance=None if method is None else tolerance
    )

    assert_eq(expected, got)

    with cudf.option_context("mode.pandas_compatible", True):
        got = gi.get_indexer(
            key, method=method, tolerance=None if method is None else tolerance
        )
    assert_eq(expected, got, check_dtype=True)
