# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import PANDAS_CURRENT_SUPPORTED_VERSION, PANDAS_VERSION
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
