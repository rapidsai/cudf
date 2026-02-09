# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("ntake", [0, 1, 123, 122, 200])
def test_series_take(ntake):
    rng = np.random.default_rng(seed=0)
    nelem = 123

    psr = pd.Series(rng.integers(0, 20, nelem))
    gsr = cudf.Series(psr)

    take_indices = rng.integers(0, len(gsr), ntake)

    actual = gsr.take(take_indices)
    expected = psr.take(take_indices)

    assert_eq(actual, expected)


def test_series_take_positional():
    psr = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])

    gsr = cudf.Series(psr)

    take_indices = [1, 2, 0, 3]

    expect = psr.take(take_indices)
    got = gsr.take(take_indices)

    assert_eq(expect, got)
