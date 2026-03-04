# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("ntake", [0, 1, 123, 122, 200])
def test_dataframe_take(ntake):
    rng = np.random.default_rng(seed=0)
    nelem = 123
    df = cudf.DataFrame(
        {
            "ii": rng.integers(0, 20, nelem),
            "ff": rng.random(nelem),
        }
    )

    take_indices = rng.integers(0, len(df), ntake)

    actual = df.take(take_indices)
    expected = df.to_pandas().take(take_indices)

    assert actual.ii.null_count == 0
    assert actual.ff.null_count == 0
    assert_eq(actual, expected)


@pytest.mark.parametrize("ntake", [1, 2, 8, 9])
def test_dataframe_take_with_multiindex(ntake):
    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame(
        {"ii": rng.integers(0, 20, 9), "ff": rng.random(9)},
        index=cudf.MultiIndex(
            levels=[["lama", "cow", "falcon"], ["speed", "weight", "length"]],
            codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        ),
    )

    take_indices = rng.integers(0, len(df), ntake)

    actual = df.take(take_indices)
    expected = df.to_pandas().take(take_indices)

    assert_eq(actual, expected)
