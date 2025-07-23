# Copyright (c) 2023-2025, NVIDIA CORPORATION.
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_construct_int_series_with_nulls_compat_mode():
    # in compatibility mode, constructing a Series
    # with nulls should result in a floating Series:
    with cudf.option_context("mode.pandas_compatible", True):
        s = cudf.Series([1, 2, None])
    assert s.dtype == np.dtype("float64")


@pytest.mark.parametrize(
    "data",
    [
        {"a": 1, "b": 2, "c": 24, "d": 1010},
        {"a": 1},
        {1: "a", 2: "b", 24: "c", 1010: "d"},
        {1: "a"},
        {"a": [1]},
    ],
)
def test_series_init_dict(data):
    pandas_series = pd.Series(data)
    cudf_series = cudf.Series(data)

    assert_eq(pandas_series, cudf_series)
