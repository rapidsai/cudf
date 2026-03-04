# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "pdi",
    [
        pd.MultiIndex(
            levels=[[1, 3.0, 4, 5], [1, 2.3, 5]],
            codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
            names=["x", "y"],
        ),
        pd.MultiIndex(
            levels=[[1, 3, 4, -10], [1, 11, 5]],
            codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
            names=["x", "y"],
        ),
        pd.MultiIndex(
            levels=[["a", "b", "c", "100"], ["1", "100", "5"]],
            codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
            names=["x", "y"],
        ),
        pytest.param(
            pd.MultiIndex(
                levels=[[None, "b", "c", "a"], ["1", None, "5"]],
                codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
                names=["x", "y"],
            ),
            marks=[
                pytest.mark.xfail(
                    reason="https://github.com/pandas-dev/pandas/issues/35584"
                )
            ],
        ),
    ],
)
@pytest.mark.parametrize("ascending", [True, False])
def test_multiindex_argsort(pdi, ascending):
    gdi = cudf.from_pandas(pdi)

    if not ascending:
        expected = pdi.argsort()[::-1]
    else:
        expected = pdi.argsort()

    actual = gdi.argsort(ascending=ascending)

    assert_eq(expected, actual)
