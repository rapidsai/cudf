# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "rangeindex",
    [
        range(np.random.default_rng(seed=0).integers(0, 100)),
        range(9, 12, 2),
        range(20, 30),
        range(100, 1000, 10),
        range(0, 10, -2),
        range(0, -10, 2),
        range(0, -10, -2),
    ],
)
@pytest.mark.parametrize(
    "func",
    ["nunique", "min", "max", "any", "values"],
)
def test_rangeindex_methods(rangeindex, func):
    gidx = cudf.RangeIndex(rangeindex)
    pidx = gidx.to_pandas()

    if func == "values":
        expected = pidx.values
        actual = gidx.values
    else:
        expected = getattr(pidx, func)()
        actual = getattr(gidx, func)()

    assert_eq(expected, actual)


def test_nunique():
    gidx = cudf.RangeIndex(5)
    pidx = pd.RangeIndex(5)

    actual = gidx.nunique()
    expected = pidx.nunique()

    assert actual == expected
