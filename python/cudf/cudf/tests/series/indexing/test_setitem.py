# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import decimal

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import PANDAS_CURRENT_SUPPORTED_VERSION, PANDAS_VERSION
from cudf.core.buffer.spill_manager import get_global_manager
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if


@pytest.mark.parametrize(
    "arg", ["b", ["a", "c"], slice(1, 2, 1), [True, False, True]]
)
def test_series_set_item(arg):
    psr = pd.Series([1, 2, 3], index=["a", "b", "c"])
    gsr = cudf.from_pandas(psr)

    psr[arg] = 11
    gsr[arg] = 11

    assert_eq(psr, gsr)


def test_column_set_unequal_length_object_by_mask():
    data = [1, 2, 3, 4, 5]
    replace_data_1 = [8, 9]
    replace_data_2 = [8, 9, 10, 11]
    mask = [True, True, False, True, False]

    psr = pd.Series(data)
    gsr = cudf.Series(data)
    assert_exceptions_equal(
        psr.__setitem__,
        gsr.__setitem__,
        ([mask, replace_data_1], {}),
        ([mask, replace_data_1], {}),
    )

    psr = pd.Series(data)
    gsr = cudf.Series(data)
    assert_exceptions_equal(
        psr.__setitem__,
        gsr.__setitem__,
        ([mask, replace_data_2], {}),
        ([mask, replace_data_2], {}),
    )


def test_categorical_setitem_invalid():
    ps = pd.Series([1, 2, 3], dtype="category")
    gs = cudf.Series([1, 2, 3], dtype="category")

    assert_exceptions_equal(
        lfunc=ps.__setitem__,
        rfunc=gs.__setitem__,
        lfunc_args_and_kwargs=([0, 5], {}),
        rfunc_args_and_kwargs=([0, 5], {}),
    )


def test_series_slice_setitem_list():
    actual = cudf.Series([[[1, 2], [2, 3]], [[3, 4]], [[4, 5]], [[6, 7]]])
    actual[slice(0, 3, 1)] = [[10, 11], [12, 23]]
    expected = cudf.Series(
        [
            [[10, 11], [12, 23]],
            [[10, 11], [12, 23]],
            [[10, 11], [12, 23]],
            [[6, 7]],
        ]
    )
    assert_eq(actual, expected)


def test_series_slice_setitem_struct():
    actual = cudf.Series(
        [
            {"a": {"b": 10}, "b": 11},
            {"a": {"b": 100}, "b": 5},
            {"a": {"b": 50}, "b": 2},
            {"a": {"b": 1000}, "b": 67},
            {"a": {"b": 4000}, "b": 1090},
        ]
    )
    actual[slice(0, 3, 1)] = {"a": {"b": 5050}, "b": 101}
    expected = cudf.Series(
        [
            {"a": {"b": 5050}, "b": 101},
            {"a": {"b": 5050}, "b": 101},
            {"a": {"b": 5050}, "b": 101},
            {"a": {"b": 1000}, "b": 67},
            {"a": {"b": 4000}, "b": 1090},
        ]
    )
    assert_eq(actual, expected)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("indices", [0, [1, 2]])
def test_series_setitem_upcasting(dtype, indices):
    sr = pd.Series([0, 0, 0], dtype=dtype)
    cr = cudf.from_pandas(sr)
    assert_eq(sr, cr)
    # Must be a non-integral floating point value that can't be losslessly
    # converted to float32, otherwise pandas will try and match the source
    # column dtype.
    new_value = np.float64(np.pi)
    col_ref = cr._column
    with expect_warning_if(dtype != np.float64):
        sr[indices] = new_value
    with expect_warning_if(dtype != np.float64):
        cr[indices] = new_value
    assert_eq(sr, cr)

    if dtype == np.float64:
        # no-op type cast should not modify backing column
        assert col_ref == cr._column


@pytest.mark.parametrize(
    "klass",
    [
        list,
        cudf.Series,
        lambda x: cudf.Series(x, index=[2, 3, 4, 5, 6]),
    ],
)
def test_series_set_equal_length_object_by_mask(klass):
    replace_data = klass([100, 200, 300, 400, 500])
    psr = pd.Series([1, 2, 3, 4, 5], dtype="Int64")
    gsr = cudf.from_pandas(psr)

    # Lengths match in trivial case
    pd_bool_col = pd.Series([True] * len(psr), dtype="boolean")
    gd_bool_col = cudf.from_pandas(pd_bool_col)
    psr[pd_bool_col] = (
        replace_data.to_pandas(nullable=True)
        if hasattr(replace_data, "to_pandas")
        else pd.Series(replace_data)
    )
    gsr[gd_bool_col] = replace_data

    assert_eq(psr.astype("float"), gsr.astype("float"))

    # Test partial masking
    psr[psr > 1] = (
        replace_data.to_pandas()
        if hasattr(replace_data, "to_pandas")
        else pd.Series(replace_data)
    )
    gsr[gsr > 1] = replace_data

    assert_eq(psr.astype("float"), gsr.astype("float"))


# TODO: these two tests could perhaps be changed once specifics of
# pandas compat wrt upcasting are decided on; this is just baking in
# status-quo.
def test_series_setitem_upcasting_string_column():
    sr = pd.Series([0, 0, 0], dtype=str)
    cr = cudf.from_pandas(sr)
    new_value = np.float64(10.5)
    sr[0] = str(new_value)
    cr[0] = str(new_value)
    assert_eq(sr, cr)


def test_series_setitem_upcasting_string_value():
    sr = cudf.Series([0, 0, 0], dtype=int)
    # This is a distinction with pandas, which lets you instead make an
    # object column with ["10", 0, 0]
    sr[0] = "10"
    assert_eq(pd.Series([10, 0, 0], dtype=int), sr)
    with pytest.raises(ValueError):
        sr[0] = "non-integer"


def test_scatter_by_slice_with_start_and_step():
    source = pd.Series([1, 2, 3, 4, 5])
    csource = cudf.from_pandas(source)
    target = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ctarget = cudf.from_pandas(target)
    target[1::2] = source
    ctarget[1::2] = csource
    assert_eq(target, ctarget)


@pytest.mark.parametrize("n", [1, 3])
def test_setitem_str_trailing_null(n):
    trailing_nulls = "\x00" * n
    s = cudf.Series(["a", "b", "c" + trailing_nulls])
    assert s[2] == "c" + trailing_nulls
    s[0] = "a" + trailing_nulls
    assert s[0] == "a" + trailing_nulls
    s[1] = trailing_nulls
    assert s[1] == trailing_nulls
    s[0] = ""
    assert s[0] == ""
    s[0] = "\x00"
    assert s[0] == "\x00"


@pytest.mark.parametrize(
    "ps",
    [
        pd.Series([1, 2, 3], index=pd.RangeIndex(0, 3)),
        pd.Series([1, 2, 3], index=pd.RangeIndex(start=2, stop=-1, step=-1)),
        pd.Series([1, 2, 3], index=pd.RangeIndex(start=1, stop=6, step=2)),
        pd.Series(
            [1, 2, 3, 4, 5], index=pd.RangeIndex(start=1, stop=-9, step=-2)
        ),
        pd.Series(
            [1, 2, 3, 4, 5], index=pd.RangeIndex(start=1, stop=-12, step=-3)
        ),
        pd.Series([1, 2, 3, 4], index=pd.RangeIndex(start=1, stop=14, step=4)),
        pd.Series(
            [1, 2, 3, 4], index=pd.RangeIndex(start=1, stop=-14, step=-4)
        ),
    ],
)
@pytest.mark.parametrize("arg", [[1], 5.6, 3.1])
def test_series_set_item_range_index(ps, arg):
    gsr = cudf.from_pandas(ps)
    psr = ps.copy(deep=True)
    psr[arg] = 11
    gsr[arg] = 11

    assert_eq(psr, gsr, check_index_type=True)


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/7448")
def test_iloc_setitem_7448():
    index = pd.MultiIndex.from_product([(1, 2), (3, 4)])
    expect = cudf.Series([1, 2, 3, 4], index=index)
    actual = cudf.from_pandas(expect)
    expect[(1, 3)] = 101
    actual[(1, 3)] = 101
    assert_eq(expect, actual)


@pytest.mark.parametrize(
    "data,item",
    [
        (
            # basic list into a list column
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [0, 0, 0],
        ),
        (
            # nested list into nested list column
            [
                [[1, 2, 3], [4, 5, 6]],
                [[1, 2, 3], [4, 5, 6]],
                [[1, 2, 3], [4, 5, 6]],
            ],
            [[0, 0, 0], [0, 0, 0]],
        ),
        (
            # NA into a list column
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            pd.NA,
        ),
        (
            # NA into nested list column
            [
                [[1, 2, 3], [4, 5, 6]],
                [[1, 2, 3], [4, 5, 6]],
                [[1, 2, 3], [4, 5, 6]],
            ],
            pd.NA,
        ),
    ],
)
def test_listcol_setitem(data, item):
    sr = cudf.Series(data)

    sr[1] = item
    data[1] = item
    expect = cudf.Series(data)

    assert_eq(expect, sr)


@pytest.mark.parametrize(
    "data,item,error_msg,error_type",
    [
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 2, 3], [4, 5, 6]],
            "Could not convert .* with type list: tried to convert to int64",
            pa.ArrowInvalid,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            0,
            "Can not set 0 into ListColumn",
            ValueError,
        ),
    ],
)
def test_listcol_setitem_error_cases(data, item, error_msg, error_type):
    sr = cudf.Series(data)
    with pytest.raises(error_type, match=error_msg):
        sr[1] = item


def test_fill_new_category():
    gs = cudf.Series(pd.Categorical(["a", "b", "c"]))
    with pytest.raises(TypeError):
        gs[0:1] = "d"


@pytest.mark.parametrize("dtype", ["int64", "float64"])
@pytest.mark.parametrize("bool_scalar", [True, False])
def test_set_bool_error(dtype, bool_scalar):
    sr = cudf.Series([1, 2, 3], dtype=dtype)
    psr = sr.to_pandas(nullable=True)

    assert_exceptions_equal(
        lfunc=sr.__setitem__,
        rfunc=psr.__setitem__,
        lfunc_args_and_kwargs=([bool_scalar],),
        rfunc_args_and_kwargs=([bool_scalar],),
    )


@pytest.mark.parametrize(
    "data", [[0, 1, 2], ["a", "b", "c"], [0.324, 32.32, 3243.23]]
)
def test_series_setitem_nat_with_non_datetimes(data):
    s = cudf.Series(data)
    with pytest.raises(TypeError):
        s[0] = cudf.NaT


def test_series_string_setitem():
    gs = cudf.Series(["abc", "def", "ghi", "xyz", "pqr"])
    ps = gs.to_pandas()

    gs[0] = "NaT"
    gs[1] = "NA"
    gs[2] = "<NA>"
    gs[3] = "NaN"

    ps[0] = "NaT"
    ps[1] = "NA"
    ps[2] = "<NA>"
    ps[3] = "NaN"

    assert_eq(gs, ps)


def test_series_error_nan_non_float_dtypes():
    s = cudf.Series(["a", "b", "c"])
    with pytest.raises(TypeError):
        s[0] = np.nan

    s = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    with pytest.raises(TypeError):
        s[0] = np.nan


def test_series_setitem_mixed_bool_dtype():
    s = cudf.Series([True, False, True])
    with pytest.raises(TypeError):
        s[0] = 10


@pytest.mark.parametrize(
    "data, item",
    [
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": cudf.NA},
                {"a": "abcde", "b": [4, 5, 6], "c": 9},
            ],
            {"a": "Hello world", "b": [], "c": cudf.NA},
        ),
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": cudf.NA},
                {"a": "abcde", "b": [4, 5, 6], "c": 9},
            ],
            {},
        ),
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": cudf.NA},
                {"a": "abcde", "b": [4, 5, 6], "c": 9},
            ],
            cudf.NA,
        ),
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": cudf.NA},
                {"a": "abcde", "b": [4, 5, 6], "c": 9},
            ],
            {"a": "Second element", "b": [1, 2], "c": 1000},
        ),
    ],
)
def test_struct_setitem(data, item):
    sr = cudf.Series(data)
    sr[1] = item
    data[1] = item
    expected = cudf.Series(data)
    assert sr.to_arrow() == expected.to_arrow()


def test_null_copy():
    col = cudf.Series(range(2049))
    col[:] = None
    assert len(col) == 2049


@pytest.mark.parametrize(
    "copy_on_write, expected",
    [
        (True, [1, 2, 3, 4, 5]),
        (False, [1, 100, 3, 4, 5]),
    ],
)
def test_series_setitem_cow(copy_on_write, expected):
    with cudf.option_context("copy_on_write", copy_on_write):
        actual = cudf.Series([1, 2, 3, 4, 5])
        new_copy = actual.copy(deep=False)

        actual[1] = 100
        assert_eq(actual, cudf.Series([1, 100, 3, 4, 5]))
        assert_eq(new_copy, cudf.Series(expected))


def test_series_setitem_both_slice_cow_on():
    with cudf.option_context("copy_on_write", True):
        actual = cudf.Series([1, 2, 3, 4, 5])
        new_copy = actual.copy(deep=False)

        actual[slice(0, 2, 1)] = 100
        assert_eq(actual, cudf.Series([100, 100, 3, 4, 5]))
        assert_eq(new_copy, cudf.Series([1, 2, 3, 4, 5]))

        new_copy[slice(2, 4, 1)] = 300
        assert_eq(actual, cudf.Series([100, 100, 3, 4, 5]))
        assert_eq(new_copy, cudf.Series([1, 2, 300, 300, 5]))


def test_series_setitem_both_slice_cow_off():
    with cudf.option_context("copy_on_write", False):
        actual = cudf.Series([1, 2, 3, 4, 5])
        new_copy = actual.copy(deep=False)

        actual[slice(0, 2, 1)] = 100
        assert_eq(actual, cudf.Series([100, 100, 3, 4, 5]))
        assert_eq(new_copy, cudf.Series([100, 100, 3, 4, 5]))

        new_copy[slice(2, 4, 1)] = 300
        assert_eq(actual, cudf.Series([100, 100, 300, 300, 5]))
        assert_eq(new_copy, cudf.Series([100, 100, 300, 300, 5]))


def test_series_setitem_partial_slice_cow_on():
    with cudf.option_context("copy_on_write", True):
        actual = cudf.Series([1, 2, 3, 4, 5])
        new_copy = actual.copy(deep=False)

        new_copy[slice(2, 4, 1)] = 300
        assert_eq(actual, cudf.Series([1, 2, 3, 4, 5]))
        assert_eq(new_copy, cudf.Series([1, 2, 300, 300, 5]))

        new_slice = actual[2:]
        assert (
            new_slice._column.base_data.owner == actual._column.base_data.owner
        )
        new_slice[0:2] = 10
        assert_eq(new_slice, cudf.Series([10, 10, 5], index=[2, 3, 4]))
        assert_eq(actual, cudf.Series([1, 2, 3, 4, 5]))


def test_series_setitem_partial_slice_cow_off():
    with cudf.option_context("copy_on_write", False):
        actual = cudf.Series([1, 2, 3, 4, 5])
        new_copy = actual.copy(deep=False)

        new_copy[slice(2, 4, 1)] = 300
        assert_eq(actual, cudf.Series([1, 2, 300, 300, 5]))
        assert_eq(new_copy, cudf.Series([1, 2, 300, 300, 5]))

        new_slice = actual[2:]
        # Since COW is off, a slice should point to the same memory
        ptr1 = new_slice._column.base_data.get_ptr(mode="read")
        ptr2 = actual._column.base_data.get_ptr(mode="read")
        assert ptr1 == ptr2

        new_slice[0:2] = 10
        assert_eq(new_slice, cudf.Series([10, 10, 5], index=[2, 3, 4]))
        assert_eq(actual, cudf.Series([1, 2, 10, 10, 5]))


def test_multiple_series_cow():
    with cudf.option_context("copy_on_write", True):
        # Verify constructing, modifying, deleting
        # multiple copies of a series preserves
        # the data appropriately when COW is enabled.
        s = cudf.Series([10, 20, 30, 40, 50])
        s1 = s.copy(deep=False)
        s2 = s.copy(deep=False)
        s3 = s.copy(deep=False)
        s4 = s2.copy(deep=False)
        s5 = s4.copy(deep=False)
        s6 = s3.copy(deep=False)

        s1[0:3] = 10000
        # s1 will be unlinked from actual data in s,
        # and then modified. Rest all should
        # contain the original data.
        assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
        for ser in [s, s2, s3, s4, s5, s6]:
            assert_eq(ser, cudf.Series([10, 20, 30, 40, 50]))

        s6[0:3] = 3000
        # s6 will be unlinked from actual data in s,
        # and then modified. Rest all should
        # contain the original data.
        assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
        assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
        for ser in [s2, s3, s4, s5]:
            assert_eq(ser, cudf.Series([10, 20, 30, 40, 50]))

        s2[1:4] = 4000
        # s2 will be unlinked from actual data in s,
        # and then modified. Rest all should
        # contain the original data.
        assert_eq(s2, cudf.Series([10, 4000, 4000, 4000, 50]))
        assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
        assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
        for ser in [s3, s4, s5]:
            assert_eq(ser, cudf.Series([10, 20, 30, 40, 50]))

        s4[2:4] = 5000
        # s4 will be unlinked from actual data in s,
        # and then modified. Rest all should
        # contain the original data.
        assert_eq(s4, cudf.Series([10, 20, 5000, 5000, 50]))
        assert_eq(s2, cudf.Series([10, 4000, 4000, 4000, 50]))
        assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
        assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
        for ser in [s3, s5]:
            assert_eq(ser, cudf.Series([10, 20, 30, 40, 50]))

        s5[2:4] = 6000
        # s5 will be unlinked from actual data in s,
        # and then modified. Rest all should
        # contain the original data.
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
        # Making a copy of s5, i.e., s7 and modifying shouldn't
        # be touching/modifying data in other series.
        assert_eq(s7, cudf.Series([10, 55, 55, 6000, 50]))

        assert_eq(s4, cudf.Series([10, 20, 5000, 5000, 50]))
        assert_eq(s2, cudf.Series([10, 4000, 4000, 4000, 50]))
        assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
        assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
        for ser in [s3]:
            assert_eq(ser, cudf.Series([10, 20, 30, 40, 50]))

        # Deleting any of the following series objects
        # shouldn't delete rest of the weekly referenced data
        # elsewhere.

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


def test_series_zero_copy_cow_on():
    with cudf.option_context("copy_on_write", True):
        s = cudf.Series([1, 2, 3, 4, 5])
        s1 = s.copy(deep=False)
        cp_array = cp.asarray(s)

        # Ensure all original data & zero-copied
        # data is same.
        assert_eq(s, cudf.Series([1, 2, 3, 4, 5]))
        assert_eq(s1, cudf.Series([1, 2, 3, 4, 5]))
        assert_eq(cp_array, cp.array([1, 2, 3, 4, 5]))

        cp_array[0:3] = 10
        # Modifying a zero-copied array should only
        # modify `s` and will leave rest of the copies
        # untouched.

        assert_eq(s.to_numpy(), np.array([10, 10, 10, 4, 5]))
        assert_eq(s, cudf.Series([10, 10, 10, 4, 5]))
        assert_eq(s1, cudf.Series([1, 2, 3, 4, 5]))
        assert_eq(cp_array, cp.array([10, 10, 10, 4, 5]))

        s2 = cudf.Series(cp_array)
        assert_eq(s2, cudf.Series([10, 10, 10, 4, 5]))

        s3 = s2.copy(deep=False)
        cp_array[0] = 20
        # Modifying a zero-copied array should modify
        # `s2` and `s` only. Because `cp_array`
        # is zero-copy shared with `s` & `s2`.

        assert_eq(s, cudf.Series([20, 10, 10, 4, 5]))
        assert_eq(s1, cudf.Series([1, 2, 3, 4, 5]))
        assert_eq(cp_array, cp.array([20, 10, 10, 4, 5]))
        assert_eq(s2, cudf.Series([20, 10, 10, 4, 5]))
        assert_eq(s3, cudf.Series([10, 10, 10, 4, 5]))

        s4 = cudf.Series([10, 20, 30, 40, 50])
        s5 = cudf.Series(s4)
        assert_eq(s5, cudf.Series([10, 20, 30, 40, 50]))
        s5[0:2] = 1
        # Modifying `s5` should also modify `s4`
        # because they are zero-copied.
        assert_eq(s5, cudf.Series([1, 1, 30, 40, 50]))
        assert_eq(s4, cudf.Series([1, 1, 30, 40, 50]))


def test_series_zero_copy_cow_off():
    is_spill_enabled = get_global_manager() is not None

    with cudf.option_context("copy_on_write", False):
        s = cudf.Series([1, 2, 3, 4, 5])
        s1 = s.copy(deep=False)
        cp_array = cp.asarray(s)

        # Ensure all original data & zero-copied
        # data is same.
        assert_eq(s, cudf.Series([1, 2, 3, 4, 5]))
        assert_eq(s1, cudf.Series([1, 2, 3, 4, 5]))
        assert_eq(cp_array, cp.array([1, 2, 3, 4, 5]))

        cp_array[0:3] = 10
        # When COW is off, modifying a zero-copied array
        # will need to modify `s` & `s1` since they are
        # shallow copied.

        assert_eq(s, cudf.Series([10, 10, 10, 4, 5]))
        assert_eq(s1, cudf.Series([10, 10, 10, 4, 5]))
        assert_eq(cp_array, cp.array([10, 10, 10, 4, 5]))

        s2 = cudf.Series(cp_array)
        assert_eq(s2, cudf.Series([10, 10, 10, 4, 5]))
        s3 = s2.copy(deep=False)
        cp_array[0] = 20

        # Modifying `cp_array`, will propagate the changes
        # across all Series objects, because they are
        # either shallow copied or zero-copied.

        assert_eq(s, cudf.Series([20, 10, 10, 4, 5]))
        assert_eq(s1, cudf.Series([20, 10, 10, 4, 5]))
        assert_eq(cp_array, cp.array([20, 10, 10, 4, 5]))
        if not is_spill_enabled:
            # Since spilling might make a copy of the data, we cannot
            # expect the two series to be a zero-copy of the cupy array
            # when spilling is enabled globally.
            assert_eq(s2, cudf.Series([20, 10, 10, 4, 5]))
            assert_eq(s3, cudf.Series([20, 10, 10, 4, 5]))

        s4 = cudf.Series([10, 20, 30, 40, 50])
        s5 = cudf.Series(s4)
        assert_eq(s5, cudf.Series([10, 20, 30, 40, 50]))
        s5[0:2] = 1

        # Modifying `s5` should also modify `s4`
        # because they are zero-copied.
        assert_eq(s5, cudf.Series([1, 1, 30, 40, 50]))
        assert_eq(s4, cudf.Series([1, 1, 30, 40, 50]))


@pytest.mark.parametrize("copy_on_write", [True, False])
def test_series_str_copy(copy_on_write):
    with cudf.option_context("copy_on_write", copy_on_write):
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


@pytest.mark.parametrize("copy_on_write", [True, False])
def test_series_cat_copy(copy_on_write):
    with cudf.option_context("copy_on_write", copy_on_write):
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


@pytest.mark.parametrize(
    "data, dtype, item, to, expect",
    [
        # scatter to a single index
        (
            ["1", "2", "3"],
            cudf.Decimal64Dtype(1, 0),
            decimal.Decimal(5),
            1,
            ["1", "5", "3"],
        ),
        (
            ["1.5", "2.5", "3.5"],
            cudf.Decimal64Dtype(2, 1),
            decimal.Decimal("5.5"),
            1,
            ["1.5", "5.5", "3.5"],
        ),
        (
            ["1.0042", "2.0042", "3.0042"],
            cudf.Decimal64Dtype(5, 4),
            decimal.Decimal("5.0042"),
            1,
            ["1.0042", "5.0042", "3.0042"],
        ),
        # scatter via boolmask
        (
            ["1", "2", "3"],
            cudf.Decimal64Dtype(1, 0),
            decimal.Decimal(5),
            [True, False, True],
            ["5", "2", "5"],
        ),
        (
            ["1.5", "2.5", "3.5"],
            cudf.Decimal64Dtype(2, 1),
            decimal.Decimal("5.5"),
            [True, True, True],
            ["5.5", "5.5", "5.5"],
        ),
        (
            ["1.0042", "2.0042", "3.0042"],
            cudf.Decimal64Dtype(5, 4),
            decimal.Decimal("5.0042"),
            [False, False, True],
            ["1.0042", "2.0042", "5.0042"],
        ),
        # We will allow assigning a decimal with less precision
        (
            ["1.00", "2.00", "3.00"],
            cudf.Decimal64Dtype(3, 2),
            decimal.Decimal(5),
            1,
            ["1.00", "5.00", "3.00"],
        ),
        # But not truncation
        (
            ["1", "2", "3"],
            cudf.Decimal64Dtype(1, 0),
            decimal.Decimal("5.5"),
            1,
            pa.ArrowInvalid,
        ),
        # We will allow for setting scalars into decimal columns
        (["1", "2", "3"], cudf.Decimal64Dtype(1, 0), 5, 1, ["1", "5", "3"]),
        # But not if it has too many digits to fit the precision
        (["1", "2", "3"], cudf.Decimal64Dtype(1, 0), 50, 1, pa.ArrowInvalid),
    ],
)
def test_series_setitem_decimal(data, dtype, item, to, expect):
    data = cudf.Series([decimal.Decimal(x) for x in data], dtype=dtype)

    if expect is pa.ArrowInvalid:
        with pytest.raises(expect):
            data[to] = item
        return
    else:
        expect = cudf.Series([decimal.Decimal(x) for x in expect], dtype=dtype)
        data[to] = item
        assert_eq(data, expect)


def test_categorical_setitem_with_nan():
    gs = cudf.Series(
        [1, 2, np.nan, 10, np.nan, None], nan_as_null=False
    ).astype("category")
    gs[[1, 3]] = np.nan

    expected_series = cudf.Series(
        [1, np.nan, np.nan, np.nan, np.nan, None], nan_as_null=False
    ).astype(gs.dtype)
    assert_eq(gs, expected_series)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize(
    "key, value",
    [
        (0, 4),
        (1, 4),
        ([0, 1], 4),
        ([0, 1], [4, 5]),
        (slice(0, 2), [4, 5]),
        (slice(1, None), [4, 5, 6, 7]),
        ([], 1),
        ([], []),
        (slice(None, None), 1),
        (slice(-1, -3), 7),
    ],
)
@pytest.mark.parametrize("nulls", ["none", "some", "all"])
def test_series_setitem_basics(key, value, nulls):
    psr = pd.Series([1, 2, 3, 4, 5])
    if nulls == "some":
        psr[[0, 4]] = None
    elif nulls == "all":
        psr[:] = None
    gsr = cudf.from_pandas(psr)
    with expect_warning_if(
        isinstance(value, list) and len(value) == 0 and nulls == "none"
    ):
        psr[key] = value
    with expect_warning_if(
        isinstance(value, list) and len(value) == 0 and not len(key) == 0
    ):
        gsr[key] = value
    assert_eq(psr, gsr, check_dtype=False)


def test_series_setitem_null():
    gsr = cudf.Series([1, 2, 3, 4])
    gsr[0] = None

    expect = cudf.Series([None, 2, 3, 4])
    got = gsr
    assert_eq(expect, got)

    gsr = cudf.Series([None, 2, 3, 4])
    gsr[0] = 1

    expect = cudf.Series([1, 2, 3, 4])
    got = gsr
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "key, value",
    [
        (0, 0.5),
        ([0, 1], 0.5),
        ([0, 1], [0.5, 2.5]),
        (slice(0, 2), [0.5, 0.25]),
    ],
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_series_setitem_dtype(key, value):
    psr = pd.Series([1, 2, 3], dtype="int32")
    gsr = cudf.from_pandas(psr)

    with pytest.warns(FutureWarning):
        psr[key] = value
    with pytest.warns(FutureWarning):
        gsr[key] = value

    assert_eq(psr, gsr)


def test_series_setitem_datetime():
    psr = pd.Series(["2001", "2002", "2003"], dtype="datetime64[ns]")
    gsr = cudf.from_pandas(psr)

    psr[0] = np.datetime64("2005")
    gsr[0] = np.datetime64("2005")

    assert_eq(psr, gsr)


def test_series_setitem_datetime_coerced():
    psr = pd.Series(["2001", "2002", "2003"], dtype="datetime64[ns]")
    gsr = cudf.from_pandas(psr)

    psr[0] = "2005"
    gsr[0] = "2005"

    assert_eq(psr, gsr)


def test_series_setitem_categorical():
    psr = pd.Series(["a", "b", "a", "c", "d"], dtype="category")
    gsr = cudf.from_pandas(psr)

    psr[0] = "d"
    gsr[0] = "d"
    assert_eq(psr, gsr)

    psr = psr.cat.add_categories(["e"])
    gsr = gsr.cat.add_categories(["e"])
    psr[0] = "e"
    gsr[0] = "e"
    assert_eq(psr, gsr)

    psr[[0, 1]] = "b"
    gsr[[0, 1]] = "b"
    assert_eq(psr, gsr)

    psr[0:3] = "e"
    gsr[0:3] = "e"
    assert_eq(psr, gsr)


@pytest.mark.parametrize(
    "key, value",
    [
        (0, "d"),
        (0, "g"),
        ([0, 1], "g"),
        ([0, 1], None),
        (slice(None, 2), "g"),
        (slice(None, 2), ["g", None]),
    ],
)
def test_series_setitem_string(key, value):
    psr = pd.Series(["a", "b", "c", "d", "e"])
    gsr = cudf.from_pandas(psr)
    psr[key] = value
    gsr[key] = value
    assert_eq(psr, gsr)

    psr = pd.Series(["a", None, "c", "d", "e"])
    gsr = cudf.from_pandas(psr)
    psr[key] = value
    gsr[key] = value
    assert_eq(psr, gsr)


def test_out_of_bounds_indexing():
    psr = pd.Series([1, 2, 3])
    gsr = cudf.from_pandas(psr)

    assert_exceptions_equal(
        lambda: psr[[0, 1, 9]],
        lambda: gsr[[0, 1, 9]],
    )
    assert_exceptions_equal(
        lambda: psr[[0, 1, -4]],
        lambda: gsr[[0, 1, -4]],
    )
    assert_exceptions_equal(
        lambda: psr.__setitem__([0, 1, 9], 2),
        lambda: gsr.__setitem__([0, 1, 9], 2),
    )
    assert_exceptions_equal(
        lambda: psr.__setitem__([0, 1, -4], 2),
        lambda: gsr.__setitem__([0, 1, -4], 2),
    )
