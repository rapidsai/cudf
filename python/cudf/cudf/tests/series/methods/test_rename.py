# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_series_unique_pandas_compatibility():
    gs = cudf.Series([10, 11, 12, 11, 10])
    ps = gs.to_pandas()
    with cudf.option_context("mode.pandas_compatible", True):
        actual = gs.unique()
    expected = ps.unique()
    assert_eq(actual, expected)


@pytest.mark.parametrize("initial_name", [None, "a"])
@pytest.mark.parametrize("name", [None, "a"])
def test_series_rename(initial_name, name):
    gsr = cudf.Series([1, 2, 3], name=initial_name)
    psr = pd.Series([1, 2, 3], name=initial_name)

    assert_eq(gsr, psr)

    actual = gsr.rename(name)
    expected = psr.rename(name)

    assert_eq(actual, expected)


@pytest.mark.parametrize("index", [lambda x: x * 2, {1: 2}])
def test_rename_index_not_supported(index):
    ser = cudf.Series(range(2))
    with pytest.raises(NotImplementedError):
        ser.rename(index=index)
