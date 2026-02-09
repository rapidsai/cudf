# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("period", [-15, -1, 0, 1, 15])
@pytest.mark.parametrize("data_empty", [False, True])
def test_shift(numeric_types_as_str, period, data_empty):
    # TODO : this function currently tests for series.shift()
    # but should instead test for dataframe.shift()
    if data_empty:
        data = None
    else:
        data = np.arange(10, dtype=numeric_types_as_str)

    gs = cudf.Series(data)
    ps = pd.Series(data)

    shifted_outcome = gs.shift(period)
    expected_outcome = ps.shift(period)

    # pandas uses NaNs to signal missing value and force converts the
    # results columns to float types
    if data_empty:
        assert_eq(
            shifted_outcome,
            expected_outcome,
            check_index_type=False,
            check_dtype=False,
        )
    else:
        assert_eq(shifted_outcome, expected_outcome, check_dtype=False)
