# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

import cudf
from cudf.api.extensions import no_default
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.testing import assert_eq
from cudf.testing._utils import (
    expect_warning_if,
)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize(
    "data",
    [
        [1, 4, 6, 1],
        np.array([1.123, 2.343, 5.890, 0.0]),
        {"a": [1.123, 2.343, np.nan, np.nan], "b": [None, 3, 9.08, None]},
    ],
)
@pytest.mark.parametrize("periods", [-5, 0, 2])
@pytest.mark.parametrize(
    "fill_method", ["ffill", "bfill", "pad", "backfill", no_default]
)
def test_dataframe_pct_change(data, periods, fill_method):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    with expect_warning_if(fill_method is not no_default):
        actual = gdf.pct_change(periods=periods, fill_method=fill_method)
    with expect_warning_if(
        fill_method is not no_default or pdf.isna().any().any()
    ):
        expected = pdf.pct_change(periods=periods, fill_method=fill_method)

    assert_eq(expected, actual)
