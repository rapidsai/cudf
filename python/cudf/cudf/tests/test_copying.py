from __future__ import division, print_function

import numpy as np
import pandas as pd
import pytest

import rmm

import cudf
import cudf._lib as libcudf
from cudf.core import Series
from cudf.core.column import column
from cudf.tests.utils import assert_eq


def test_gather_single_col():
    col = column.as_column(np.arange(100), dtype=np.int32)
    gather_map = np.array([0, 1, 2, 3, 5, 8, 13, 21], dtype=np.int32)

    device_gather_map = rmm.to_device(gather_map)

    out = libcudf.copying.gather(col, device_gather_map)

    np.testing.assert_array_equal(out.to_array(), gather_map)


def test_gather_cols():
    cols = [
        column.as_column(np.arange(10), dtype=np.int32),
        column.as_column(np.arange(0.0, 2.0, 0.2), dtype=np.float32),
    ]
    gather_map = np.array([0, 1, 2, 3, 5, 8], dtype=np.int32)

    expected = np.array(gather_map * 0.2, dtype=np.float32)

    device_gather_map = rmm.to_device(gather_map)

    out = libcudf.copying.gather(cols, device_gather_map)

    np.testing.assert_array_equal(out[0].to_array(), gather_map)
    np.testing.assert_array_almost_equal(out[1].to_array(), expected)


def test_gather_string_col():
    col = column.as_column(["a", "b", "c", "d"])
    gather_map = column.as_column([0, 2, 3], dtype="int32").data_array_view
    result = libcudf.copying.gather(col, gather_map)
    assert result.data_array_view.to_host() == ["a", "c", "d"]

    col = column.as_column(["a", "b", None, "d"])
    gather_map = column.as_column([0, 2, 3], dtype="int32").data_array_view
    result = libcudf.copying.gather(col, gather_map)
    assert result.data_array_view.to_host() == ["a", None, "d"]


@pytest.mark.parametrize(
    "dtype",
    ["bool", "int8", "int16", "int32", "int64", "float32", "float64", "str"],
)
def test_repeat(dtype):
    arr = np.random.rand(10) * 10
    repeats = np.random.randint(10, size=10)
    psr = pd.Series(arr).astype(dtype)
    gsr = cudf.from_pandas(psr)

    assert_eq(psr.repeat(repeats), gsr.repeat(repeats))


def test_repeat_index():
    arrays = [[1, 1, 2, 2], ["red", "blue", "red", "blue"]]
    psr = pd.MultiIndex.from_arrays(arrays, names=("number", "color"))
    gsr = cudf.from_pandas(psr)
    repeats = np.random.randint(10, size=4)

    assert_eq(psr.repeat(repeats), gsr.repeat(repeats))


def test_repeat_dataframe():
    psr = pd.DataFrame({"a": [1, 1, 2, 2]})
    gsr = cudf.from_pandas(psr)
    repeats = np.random.randint(10, size=4)

    # pd.DataFrame doesn't have repeat() so as a workaround, we are
    # comparing pd.Series.repeat() with cudf.DataFrame.repeat()['a']
    assert_eq(psr["a"].repeat(repeats), gsr.repeat(repeats)["a"])


@pytest.mark.parametrize(
    "dtype", ["bool", "int8", "int16", "int32", "int64", "float32", "float64"]
)
def test_repeat_scalar(dtype):
    arr = np.random.rand(10) * 10
    repeats = 10
    psr = pd.Series(arr).astype(dtype)
    gsr = cudf.from_pandas(psr)

    assert_eq(psr.repeat(repeats), gsr.repeat(repeats))


def test_null_copy():
    col = Series(np.arange(2049))
    col[:] = None
    assert len(col) == 2049
