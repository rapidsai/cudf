# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import (
    expect_warning_if,
)


@pytest.mark.parametrize(
    "data",
    [
        [0.25, 0.5, 0.2, -0.05],
        [0, 1, 2, np.nan, 4, cudf.NA, 6],
    ],
)
@pytest.mark.parametrize("lag", [1, 4])
def test_autocorr(data, lag):
    cudf_series = cudf.Series(data)
    psr = cudf_series.to_pandas()

    cudf_corr = cudf_series.autocorr(lag=lag)

    # autocorrelation is undefined (nan) for less than two entries, but pandas
    # short-circuits when there are 0 entries and bypasses the numpy function
    # call that generates an error.
    num_both_valid = (psr.notna() & psr.shift(lag).notna()).sum()
    with expect_warning_if(num_both_valid == 1, RuntimeWarning):
        pd_corr = psr.autocorr(lag=lag)

    assert_eq(pd_corr, cudf_corr)
