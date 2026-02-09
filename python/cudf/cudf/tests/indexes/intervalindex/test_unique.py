# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_interval_index_unique():
    interval_list = [
        np.nan,
        pd.Interval(2.0, 3.0, closed="right"),
        pd.Interval(3.0, 4.0, closed="right"),
        np.nan,
        pd.Interval(3.0, 4.0, closed="right"),
        pd.Interval(3.0, 4.0, closed="right"),
    ]
    pi = pd.Index(interval_list)
    gi = cudf.from_pandas(pi)

    expected = pi.unique()
    actual = gi.unique()

    assert_eq(expected, actual)
