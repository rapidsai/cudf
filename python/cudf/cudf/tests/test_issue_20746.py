# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

import cudf
from cudf.testing import assert_series_equal


def test_cast_float_nan_to_bool_pandas_compat():
    """
    Regression test for Issue #20746.
    Ensures that casting float columns with NaNs to boolean
    treats NaNs as True (matching Pandas behavior) when
    mode.pandas_compatible is enabled.
    """
    # Enable pandas compatibility mode
    cudf.set_option("mode.pandas_compatible", True)

    try:
        data = [1.0, 0.0, np.nan, None]

        # Create cuDF Series
        gs = cudf.Series(data, dtype="float64")

        # Cast to bool
        got = gs.astype("bool")

        # Create expected Pandas Series (Pandas casts NaN/None to True)
        expected = pd.Series([True, False, True, True], dtype="bool")

        # Verify
        # In Pandas compat mode, we expect NO nulls in the boolean result
        assert got.null_count == 0

        # Convert to pandas for easy comparison or use testing utils
        expected_cudf = cudf.Series(expected)

        assert_series_equal(got, expected_cudf)

    finally:
        # Reset option to avoid side effects on other tests
        cudf.set_option("mode.pandas_compatible", False)
