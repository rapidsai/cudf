# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

import cudf
from cudf.api.extensions import no_default
from cudf.core._compat import PANDAS_CURRENT_SUPPORTED_VERSION, PANDAS_VERSION
from cudf.testing._utils import expect_warning_if


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
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
@pytest.mark.parametrize(
    "fill_method", ["ffill", "bfill", "pad", "backfill", no_default, None]
)
def test_series_pct_change(data, periods, fill_method):
    cs = cudf.Series(data)
    ps = cs.to_pandas()

    if np.abs(periods) <= len(cs):
        with expect_warning_if(fill_method not in (no_default, None)):
            got = cs.pct_change(periods=periods, fill_method=fill_method)
        with expect_warning_if(
            (
                fill_method not in (no_default, None)
                or (fill_method is not None and ps.isna().any())
            )
        ):
            expected = ps.pct_change(periods=periods, fill_method=fill_method)
        np.testing.assert_array_almost_equal(
            got.to_numpy(na_value=np.nan), expected
        )
