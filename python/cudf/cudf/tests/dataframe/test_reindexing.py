# Copyright (c) 2023, NVIDIA CORPORATION.
import numpy as np

import cudf
from cudf.testing._utils import assert_eq


def test_reindexing_int_series_compat_mode():
    # in compatibility mode, reindexing an int Series
    # should result in a floating Series:
    with cudf.option_context("mode.pandas_compatible", True):
        s = cudf.Series([1, 2]).reindex([1, 2, 3])
    expected = cudf.Series([2, np.nan, np.nan], index=[1, 2, 3])
    assert s.dtype == np.dtype("float64")
    assert assert_eq(expected, s)
