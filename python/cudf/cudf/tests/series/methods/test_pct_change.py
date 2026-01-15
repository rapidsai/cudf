# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

import cudf


@pytest.mark.parametrize(
    "data",
    [
        np.random.default_rng(seed=0).normal(-100, 100, 1000),
        np.random.default_rng(seed=0).integers(-50, 50, 1000),
        np.zeros(100),
        np.array([1.123, 2.343, np.nan, 0.0]),
        np.array([-2, 3.75, 6, None, None, None, -8.5, None, 4.2]),
        np.array([], dtype="float64"),
        np.array([-3]),
    ],
)
@pytest.mark.parametrize("periods", [-5, 0, 5])
def test_series_pct_change(data, periods):
    cs = cudf.Series(data)
    ps = cs.to_pandas()

    got = cs.pct_change(periods=periods)
    expected = ps.pct_change(periods=periods)
    np.testing.assert_array_almost_equal(
        got.to_numpy(na_value=np.nan), expected
    )
