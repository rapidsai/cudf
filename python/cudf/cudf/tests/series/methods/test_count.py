# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


def test_series_count_invalid_param():
    s = cudf.Series([], dtype="float64")
    with pytest.raises(TypeError):
        s.count(skipna=True)


def test_series_dataframe_count_float():
    gs = cudf.Series([1, 2, 3, None, np.nan, 10], nan_as_null=False)
    ps = cudf.Series([1, 2, 3, None, np.nan, 10])

    with cudf.option_context("mode.pandas_compatible", True):
        assert_eq(ps.count(), gs.count())
        assert_eq(ps.to_frame().count(), gs.to_frame().count())
    with cudf.option_context("mode.pandas_compatible", False):
        assert_eq(gs.count(), gs.to_pandas(nullable=True).count())
        assert_eq(
            gs.to_frame().count(),
            gs.to_frame().to_pandas(nullable=True).count(),
        )
