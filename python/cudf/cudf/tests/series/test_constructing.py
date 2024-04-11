# Copyright (c) 2023, NVIDIA CORPORATION.
import numpy as np

import cudf


def test_construct_int_series_with_nulls_compat_mode():
    # in compatibility mode, constructing a Series
    # with nulls should result in a floating Series:
    with cudf.option_context("mode.pandas_compatible", True):
        s = cudf.Series([1, 2, None])
    assert s.dtype == np.dtype("float64")
