# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_dropna_multiindex(dropna_how):
    pi = pd.MultiIndex.from_arrays([[1, None, 2], [None, None, 2]])
    gi = cudf.from_pandas(pi)

    expect = pi.dropna(dropna_how)
    got = gi.dropna(dropna_how)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        [
            [pd.Timestamp("2020-01-01"), pd.NaT, pd.Timestamp("2020-02-01")],
            [pd.NaT, pd.NaT, pd.Timestamp("2020-03-01")],
        ],
        [
            [pd.Timestamp("2020-01-01"), pd.NaT, pd.Timestamp("2020-02-01")],
            [np.nan, np.nan, 1.0],
        ],
        [[1.0, np.nan, 2.0], [np.nan, np.nan, 1.0]],
    ],
)
def test_dropna_multiindex_2(data, dropna_how):
    pi = pd.MultiIndex.from_arrays(data)
    gi = cudf.from_pandas(pi)

    expect = pi.dropna(dropna_how)
    got = gi.dropna(dropna_how)

    assert_eq(expect, got)
