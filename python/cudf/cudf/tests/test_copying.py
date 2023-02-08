# Copyright (c) 2020-2023, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import Series
from cudf.testing._utils import NUMERIC_TYPES, OTHER_TYPES, assert_eq


@pytest.mark.parametrize("dtype", NUMERIC_TYPES + OTHER_TYPES)
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


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
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


# TODO: Make use of set_option context manager
# once https://github.com/rapidsai/cudf/issues/12736
# is resolved.


def test_series_setitem_cow_on():
    original_cow_setting = cudf.get_option("copy_on_write")
    cudf.set_option("copy_on_write", True)
    actual = cudf.Series([1, 2, 3, 4, 5])
    new_copy = actual.copy(deep=False)

    actual[1] = 100
    assert_eq(actual, cudf.Series([1, 100, 3, 4, 5]))
    assert_eq(new_copy, cudf.Series([1, 2, 3, 4, 5]))

    actual = cudf.Series([1, 2, 3, 4, 5])
    new_copy = actual.copy(deep=False)

    actual[slice(0, 2, 1)] = 100
    assert_eq(actual, cudf.Series([100, 100, 3, 4, 5]))
    assert_eq(new_copy, cudf.Series([1, 2, 3, 4, 5]))

    new_copy[slice(2, 4, 1)] = 300
    assert_eq(actual, cudf.Series([100, 100, 3, 4, 5]))
    assert_eq(new_copy, cudf.Series([1, 2, 300, 300, 5]))

    actual = cudf.Series([1, 2, 3, 4, 5])
    new_copy = actual.copy(deep=False)

    new_copy[slice(2, 4, 1)] = 300
    assert_eq(actual, cudf.Series([1, 2, 3, 4, 5]))
    assert_eq(new_copy, cudf.Series([1, 2, 300, 300, 5]))

    new_slice = actual[2:]
    assert new_slice._column.base_data._ptr == actual._column.base_data._ptr
    new_slice[0:2] = 10
    assert_eq(new_slice, cudf.Series([10, 10, 5], index=[2, 3, 4]))
    assert_eq(actual, cudf.Series([1, 2, 3, 4, 5]))
    cudf.set_option("copy_on_write", original_cow_setting)


def test_series_setitem_cow_off():
    original_cow_setting = cudf.get_option("copy_on_write")
    cudf.set_option("copy_on_write", False)
    actual = cudf.Series([1, 2, 3, 4, 5])
    new_copy = actual.copy(deep=False)

    actual[1] = 100
    assert_eq(actual, cudf.Series([1, 100, 3, 4, 5]))
    assert_eq(new_copy, cudf.Series([1, 100, 3, 4, 5]))

    actual = cudf.Series([1, 2, 3, 4, 5])
    new_copy = actual.copy(deep=False)

    actual[slice(0, 2, 1)] = 100
    assert_eq(actual, cudf.Series([100, 100, 3, 4, 5]))
    assert_eq(new_copy, cudf.Series([100, 100, 3, 4, 5]))

    new_copy[slice(2, 4, 1)] = 300
    assert_eq(actual, cudf.Series([100, 100, 300, 300, 5]))
    assert_eq(new_copy, cudf.Series([100, 100, 300, 300, 5]))

    actual = cudf.Series([1, 2, 3, 4, 5])
    new_copy = actual.copy(deep=False)

    new_copy[slice(2, 4, 1)] = 300
    assert_eq(actual, cudf.Series([1, 2, 300, 300, 5]))
    assert_eq(new_copy, cudf.Series([1, 2, 300, 300, 5]))

    new_slice = actual[2:]
    assert new_slice._column.base_data._ptr == actual._column.base_data._ptr
    new_slice[0:2] = 10
    assert_eq(new_slice, cudf.Series([10, 10, 5], index=[2, 3, 4]))
    assert_eq(actual, cudf.Series([1, 2, 10, 10, 5]))
    cudf.set_option("copy_on_write", original_cow_setting)


def test_multiple_series_cow():
    original_cow_setting = cudf.get_option("copy_on_write")
    cudf.set_option("copy_on_write", True)
    s = cudf.Series([10, 20, 30, 40, 50])
    s1 = s.copy(deep=False)
    s2 = s.copy(deep=False)
    s3 = s.copy(deep=False)
    s4 = s2.copy(deep=False)
    s5 = s4.copy(deep=False)
    s6 = s3.copy(deep=False)

    s1[0:3] = 10000
    assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
    for ser in [s, s2, s3, s4, s5, s6]:
        assert_eq(ser, cudf.Series([10, 20, 30, 40, 50]))

    s6[0:3] = 3000
    assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
    assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
    for ser in [s2, s3, s4, s5]:
        assert_eq(ser, cudf.Series([10, 20, 30, 40, 50]))

    s2[1:4] = 4000
    assert_eq(s2, cudf.Series([10, 4000, 4000, 4000, 50]))
    assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
    assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
    for ser in [s3, s4, s5]:
        assert_eq(ser, cudf.Series([10, 20, 30, 40, 50]))

    s4[2:4] = 5000
    assert_eq(s4, cudf.Series([10, 20, 5000, 5000, 50]))
    assert_eq(s2, cudf.Series([10, 4000, 4000, 4000, 50]))
    assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
    assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
    for ser in [s3, s5]:
        assert_eq(ser, cudf.Series([10, 20, 30, 40, 50]))

    s5[2:4] = 6000
    assert_eq(s5, cudf.Series([10, 20, 6000, 6000, 50]))
    assert_eq(s4, cudf.Series([10, 20, 5000, 5000, 50]))
    assert_eq(s2, cudf.Series([10, 4000, 4000, 4000, 50]))
    assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
    assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
    for ser in [s3]:
        assert_eq(ser, cudf.Series([10, 20, 30, 40, 50]))

    s7 = s5.copy(deep=False)
    assert_eq(s7, cudf.Series([10, 20, 6000, 6000, 50]))
    s7[1:3] = 55
    assert_eq(s7, cudf.Series([10, 55, 55, 6000, 50]))

    assert_eq(s4, cudf.Series([10, 20, 5000, 5000, 50]))
    assert_eq(s2, cudf.Series([10, 4000, 4000, 4000, 50]))
    assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
    assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
    for ser in [s3]:
        assert_eq(ser, cudf.Series([10, 20, 30, 40, 50]))

    del s2

    assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
    assert_eq(s3, cudf.Series([10, 20, 30, 40, 50]))
    assert_eq(s4, cudf.Series([10, 20, 5000, 5000, 50]))
    assert_eq(s5, cudf.Series([10, 20, 6000, 6000, 50]))
    assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
    assert_eq(s7, cudf.Series([10, 55, 55, 6000, 50]))

    del s4
    del s1

    assert_eq(s3, cudf.Series([10, 20, 30, 40, 50]))
    assert_eq(s5, cudf.Series([10, 20, 6000, 6000, 50]))
    assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
    assert_eq(s7, cudf.Series([10, 55, 55, 6000, 50]))

    del s
    del s6

    assert_eq(s3, cudf.Series([10, 20, 30, 40, 50]))
    assert_eq(s5, cudf.Series([10, 20, 6000, 6000, 50]))
    assert_eq(s7, cudf.Series([10, 55, 55, 6000, 50]))

    del s5

    assert_eq(s3, cudf.Series([10, 20, 30, 40, 50]))
    assert_eq(s7, cudf.Series([10, 55, 55, 6000, 50]))

    del s3
    assert_eq(s7, cudf.Series([10, 55, 55, 6000, 50]))
    cudf.set_option("copy_on_write", original_cow_setting)


def test_series_zero_copy_cow_on():
    original_cow_setting = cudf.get_option("copy_on_write")
    cudf.set_option("copy_on_write", True)
    s = cudf.Series([1, 2, 3, 4, 5])
    s1 = s.copy(deep=False)
    cp_array = cp.asarray(s)

    assert_eq(s, cudf.Series([1, 2, 3, 4, 5]))
    assert_eq(s1, cudf.Series([1, 2, 3, 4, 5]))
    assert_eq(cp_array, cp.array([1, 2, 3, 4, 5]))

    cp_array[0:3] = 10

    assert_eq(s, cudf.Series([10, 10, 10, 4, 5]))
    assert_eq(s1, cudf.Series([1, 2, 3, 4, 5]))
    assert_eq(cp_array, cp.array([10, 10, 10, 4, 5]))

    s2 = cudf.Series(cp_array)
    assert_eq(s2, cudf.Series([10, 10, 10, 4, 5]))
    s3 = s2.copy(deep=False)
    cp_array[0] = 20

    assert_eq(s, cudf.Series([20, 10, 10, 4, 5]))
    assert_eq(s1, cudf.Series([1, 2, 3, 4, 5]))
    assert_eq(cp_array, cp.array([20, 10, 10, 4, 5]))
    assert_eq(s2, cudf.Series([20, 10, 10, 4, 5]))
    assert_eq(s3, cudf.Series([10, 10, 10, 4, 5]))

    s4 = cudf.Series([10, 20, 30, 40, 50])
    s5 = cudf.Series(s4)
    assert_eq(s5, cudf.Series([10, 20, 30, 40, 50]))
    s5[0:2] = 1
    assert_eq(s5, cudf.Series([1, 1, 30, 40, 50]))
    assert_eq(s4, cudf.Series([1, 1, 30, 40, 50]))
    cudf.set_option("copy_on_write", original_cow_setting)


def test_series_zero_copy_cow_off():
    original_cow_setting = cudf.get_option("copy_on_write")
    cudf.set_option("copy_on_write", False)
    s = cudf.Series([1, 2, 3, 4, 5])
    s1 = s.copy(deep=False)
    cp_array = cp.asarray(s)

    assert_eq(s, cudf.Series([1, 2, 3, 4, 5]))
    assert_eq(s1, cudf.Series([1, 2, 3, 4, 5]))
    assert_eq(cp_array, cp.array([1, 2, 3, 4, 5]))

    cp_array[0:3] = 10

    assert_eq(s, cudf.Series([10, 10, 10, 4, 5]))
    assert_eq(s1, cudf.Series([10, 10, 10, 4, 5]))
    assert_eq(cp_array, cp.array([10, 10, 10, 4, 5]))

    s2 = cudf.Series(cp_array)
    assert_eq(s2, cudf.Series([10, 10, 10, 4, 5]))
    s3 = s2.copy(deep=False)
    cp_array[0] = 20

    assert_eq(s, cudf.Series([20, 10, 10, 4, 5]))
    assert_eq(s1, cudf.Series([20, 10, 10, 4, 5]))
    assert_eq(cp_array, cp.array([20, 10, 10, 4, 5]))
    assert_eq(s2, cudf.Series([20, 10, 10, 4, 5]))
    assert_eq(s3, cudf.Series([20, 10, 10, 4, 5]))

    s4 = cudf.Series([10, 20, 30, 40, 50])
    s5 = cudf.Series(s4)
    assert_eq(s5, cudf.Series([10, 20, 30, 40, 50]))
    s5[0:2] = 1
    assert_eq(s5, cudf.Series([1, 1, 30, 40, 50]))
    assert_eq(s4, cudf.Series([1, 1, 30, 40, 50]))
    cudf.set_option("copy_on_write", original_cow_setting)


@pytest.mark.parametrize("copy_on_write", [True, False])
def test_series_str_copy(copy_on_write):
    original_cow_setting = cudf.get_option("copy_on_write")
    cudf.set_option("copy_on_write", copy_on_write)
    s = cudf.Series(["a", "b", "c", "d", "e"])
    s1 = s.copy(deep=True)
    s2 = s.copy(deep=True)

    assert_eq(s, cudf.Series(["a", "b", "c", "d", "e"]))
    assert_eq(s1, cudf.Series(["a", "b", "c", "d", "e"]))
    assert_eq(s2, cudf.Series(["a", "b", "c", "d", "e"]))

    s[0:3] = "abc"

    assert_eq(s, cudf.Series(["abc", "abc", "abc", "d", "e"]))
    assert_eq(s1, cudf.Series(["a", "b", "c", "d", "e"]))
    assert_eq(s2, cudf.Series(["a", "b", "c", "d", "e"]))

    s2[1:4] = "xyz"

    assert_eq(s, cudf.Series(["abc", "abc", "abc", "d", "e"]))
    assert_eq(s1, cudf.Series(["a", "b", "c", "d", "e"]))
    assert_eq(s2, cudf.Series(["a", "xyz", "xyz", "xyz", "e"]))
    cudf.set_option("copy_on_write", original_cow_setting)


@pytest.mark.parametrize("copy_on_write", [True, False])
def test_series_cat_copy(copy_on_write):
    original_cow_setting = cudf.get_option("copy_on_write")
    cudf.set_option("copy_on_write", copy_on_write)
    s = cudf.Series([10, 20, 30, 40, 50], dtype="category")
    s1 = s.copy(deep=True)
    s2 = s1.copy(deep=True)
    s3 = s1.copy(deep=True)

    s[0] = 50
    assert_eq(s, cudf.Series([50, 20, 30, 40, 50], dtype=s.dtype))
    assert_eq(s1, cudf.Series([10, 20, 30, 40, 50], dtype="category"))
    assert_eq(s2, cudf.Series([10, 20, 30, 40, 50], dtype="category"))
    assert_eq(s3, cudf.Series([10, 20, 30, 40, 50], dtype="category"))

    s2[3] = 10
    s3[2:5] = 20
    assert_eq(s, cudf.Series([50, 20, 30, 40, 50], dtype=s.dtype))
    assert_eq(s1, cudf.Series([10, 20, 30, 40, 50], dtype=s.dtype))
    assert_eq(s2, cudf.Series([10, 20, 30, 10, 50], dtype=s.dtype))
    assert_eq(s3, cudf.Series([10, 20, 20, 20, 20], dtype=s.dtype))
    cudf.set_option("copy_on_write", original_cow_setting)


def test_dataframe_cow_slice_setitem():
    original_cow_setting = cudf.get_option("copy_on_write")
    cudf.set_option("copy_on_write", True)
    df = cudf.DataFrame({"a": [10, 11, 12, 13, 14], "b": [20, 30, 40, 50, 60]})
    slice_df = df[1:4]

    assert_eq(
        slice_df,
        cudf.DataFrame(
            {"a": [11, 12, 13], "b": [30, 40, 50]}, index=[1, 2, 3]
        ),
    )

    slice_df["a"][2] = 1111

    assert_eq(
        slice_df,
        cudf.DataFrame(
            {"a": [11, 1111, 13], "b": [30, 40, 50]}, index=[1, 2, 3]
        ),
    )
    assert_eq(
        df,
        cudf.DataFrame({"a": [10, 11, 12, 13, 14], "b": [20, 30, 40, 50, 60]}),
    )
    cudf.set_option("copy_on_write", original_cow_setting)
