# Copyright (c) 2018-2024, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_CURRENT_SUPPORTED_VERSION, PANDAS_VERSION
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if


@pytest.mark.parametrize("df", [pd.DataFrame({"a": [1, 2, 3]})])
@pytest.mark.parametrize("arg", [[True, False, True], [True, True, True]])
@pytest.mark.parametrize("value", [0, -1])
def test_dataframe_setitem_bool_mask_scaler(df, arg, value):
    gdf = cudf.from_pandas(df)

    df[arg] = value
    gdf[arg] = value
    assert_eq(df, gdf)


def test_dataframe_setitem_scaler_bool():
    df = pd.DataFrame({"a": [1, 2, 3]})
    df[[True, False, True]] = pd.DataFrame({"a": [-1, -2]})

    gdf = cudf.DataFrame({"a": [1, 2, 3]})
    gdf[[True, False, True]] = cudf.DataFrame({"a": [-1, -2]})
    assert_eq(df, gdf)


@pytest.mark.parametrize(
    "df",
    [pd.DataFrame({"a": [1, 2, 3]}), pd.DataFrame({"a": ["x", "y", "z"]})],
)
@pytest.mark.parametrize("arg", [["a"], "a", "b"])
@pytest.mark.parametrize(
    "value", [-10, pd.DataFrame({"a": [-1, -2, -3]}), "abc"]
)
def test_dataframe_setitem_columns(df, arg, value):
    gdf = cudf.from_pandas(df)
    cudf_replace_value = value

    if isinstance(cudf_replace_value, pd.DataFrame):
        cudf_replace_value = cudf.from_pandas(value)

    df[arg] = value
    gdf[arg] = cudf_replace_value
    assert_eq(df, gdf, check_dtype=False)


@pytest.mark.parametrize("df", [pd.DataFrame({"a": [1, 2, 3]})])
@pytest.mark.parametrize("arg", [["b", "c"]])
@pytest.mark.parametrize(
    "value",
    [
        pd.DataFrame({"0": [-1, -2, -3], "1": [-0, -10, -1]}),
        10,
        20,
        30,
        "rapids",
        "ai",
        0.32234,
        np.datetime64(1324232423423342, "ns"),
        np.timedelta64(34234324234324234, "ns"),
    ],
)
def test_dataframe_setitem_new_columns(df, arg, value):
    gdf = cudf.from_pandas(df)
    cudf_replace_value = value

    if isinstance(cudf_replace_value, pd.DataFrame):
        cudf_replace_value = cudf.from_pandas(value)

    df[arg] = value
    gdf[arg] = cudf_replace_value
    assert_eq(df, gdf, check_dtype=True)


# set_item_series inconsistency
def test_series_setitem_index():
    df = pd.DataFrame(
        data={"b": [-1, -2, -3], "c": [1, 2, 3]}, index=[1, 2, 3]
    )

    df["b"] = pd.Series(data=[12, 11, 10], index=[3, 2, 1])
    gdf = cudf.DataFrame(
        data={"b": [-1, -2, -3], "c": [1, 2, 3]}, index=[1, 2, 3]
    )
    gdf["b"] = cudf.Series(data=[12, 11, 10], index=[3, 2, 1])
    assert_eq(df, gdf, check_dtype=False)


@pytest.mark.parametrize("psr", [pd.Series([1, 2, 3], index=["a", "b", "c"])])
@pytest.mark.parametrize(
    "arg", ["b", ["a", "c"], slice(1, 2, 1), [True, False, True]]
)
def test_series_set_item(psr, arg):
    gsr = cudf.from_pandas(psr)

    psr[arg] = 11
    gsr[arg] = 11

    assert_eq(psr, gsr)


def test_series_setitem_singleton_range():
    sr = cudf.Series([1, 2, 3], dtype=np.int64)
    psr = sr.to_pandas()
    value = np.asarray([7], dtype=np.int64)
    sr.iloc[:1] = value
    psr.iloc[:1] = value
    assert_eq(sr, cudf.Series([7, 2, 3], dtype=np.int64))
    assert_eq(sr, psr, check_dtype=True)


@pytest.mark.xfail(reason="Copy-on-Write should make a copy")
@pytest.mark.parametrize(
    "index",
    [
        pd.MultiIndex.from_frame(
            pd.DataFrame({"b": [3, 2, 1], "c": ["a", "b", "c"]})
        ),
        ["a", "b", "c"],
    ],
)
def test_setitem_dataframe_series_inplace(index):
    gdf = cudf.DataFrame({"a": [1, 2, 3]}, index=index)
    expected = gdf.copy()
    with cudf.option_context("copy_on_write", True):
        gdf["a"].replace(1, 500, inplace=True)

    assert_eq(expected, gdf)


@pytest.mark.parametrize(
    "replace_data",
    [
        [100, 200, 300, 400, 500],
        cudf.Series([100, 200, 300, 400, 500]),
        cudf.Series([100, 200, 300, 400, 500], index=[2, 3, 4, 5, 6]),
    ],
)
def test_series_set_equal_length_object_by_mask(replace_data):
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


def test_column_set_equal_length_object_by_mask():
    # Series.__setitem__ might bypass some of the cases
    # handled in column.__setitem__ so this test is needed

    data = cudf.Series([0, 0, 1, 1, 1])._column
    replace_data = cudf.Series([100, 200, 300, 400, 500])._column
    bool_col = cudf.Series([True, True, True, True, True])._column

    data[bool_col] = replace_data
    assert_eq(
        cudf.Series._from_column(data),
        cudf.Series._from_column(replace_data),
    )

    data = cudf.Series([0, 0, 1, 1, 1])._column
    bool_col = cudf.Series([True, False, True, False, True])._column
    data[bool_col] = replace_data

    assert_eq(
        cudf.Series._from_column(data),
        cudf.Series([100, 0, 300, 1, 500]),
    )


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

    actual = cudf.Series([[[1, 2], [2, 3]], [[3, 4]], [[4, 5]], [[6, 7]]])
    actual[0:3] = cudf.Scalar([[10, 11], [12, 23]])

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

    actual = cudf.Series(
        [
            {"a": {"b": 10}, "b": 11},
            {"a": {"b": 100}, "b": 5},
            {"a": {"b": 50}, "b": 2},
            {"a": {"b": 1000}, "b": 67},
            {"a": {"b": 4000}, "b": 1090},
        ]
    )
    actual[0:3] = cudf.Scalar({"a": {"b": 5050}, "b": 101})

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


# TODO: these two tests could perhaps be changed once specifics of
# pandas compat wrt upcasting are decided on; this is just baking in
# status-quo.
def test_series_setitem_upcasting_string_column():
    sr = pd.Series([0, 0, 0], dtype=str)
    cr = cudf.from_pandas(sr)
    new_value = np.float64(10.5)
    sr[0] = str(new_value)
    cr[0] = new_value
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


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/7448")
def test_iloc_setitem_7448():
    index = pd.MultiIndex.from_product([(1, 2), (3, 4)])
    expect = cudf.Series([1, 2, 3, 4], index=index)
    actual = cudf.from_pandas(expect)
    expect[(1, 3)] = 101
    actual[(1, 3)] = 101
    assert_eq(expect, actual)


@pytest.mark.parametrize(
    "value",
    [
        "7",
        pytest.param(
            ["7", "8"],
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/11298"
            ),
        ),
    ],
)
def test_loc_setitem_string_11298(value):
    df = pd.DataFrame({"a": ["a", "b", "c"]})
    cdf = cudf.from_pandas(df)

    df.loc[:1, "a"] = value

    cdf.loc[:1, "a"] = value

    assert_eq(df, cdf)


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/11944")
def test_loc_setitem_list_11944():
    df = pd.DataFrame(
        data={"a": ["yes", "no"], "b": [["l1", "l2"], ["c", "d"]]}
    )
    cdf = cudf.from_pandas(df)
    df.loc[df.a == "yes", "b"] = [["hello"]]
    cdf.loc[df.a == "yes", "b"] = [["hello"]]
    assert_eq(df, cdf)


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/12504")
def test_loc_setitem_extend_empty_12504():
    df = pd.DataFrame(columns=["a"])
    cdf = cudf.from_pandas(df)

    df.loc[0] = [1]

    cdf.loc[0] = [1]

    assert_eq(df, cdf)


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/12505")
def test_loc_setitem_extend_existing_12505():
    df = pd.DataFrame({"a": [0]})
    cdf = cudf.from_pandas(df)

    df.loc[1] = 1

    cdf.loc[1] = 1

    assert_eq(df, cdf)


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/12801")
def test_loc_setitem_add_column_partial_12801():
    df = pd.DataFrame({"a": [0, 1, 2]})
    cdf = cudf.from_pandas(df)

    df.loc[df.a < 2, "b"] = 1

    cdf.loc[cdf.a < 2, "b"] = 1

    assert_eq(df, cdf)


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/13031")
@pytest.mark.parametrize("other_index", [["1", "3", "2"], [1, 2, 3]])
def test_loc_setitem_series_index_alignment_13031(other_index):
    s = pd.Series([1, 2, 3], index=["1", "2", "3"])
    other = pd.Series([5, 6, 7], index=other_index)

    cs = cudf.from_pandas(s)
    cother = cudf.from_pandas(other)

    s.loc[["1", "3"]] = other

    cs.loc[["1", "3"]] = cother

    assert_eq(s, cs)


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
@pytest.mark.parametrize("arg", [*list(range(-20, 20)), 5.6, 3.1])
def test_series_set_item_range_index(ps, arg):
    gsr = cudf.from_pandas(ps)
    psr = ps.copy(deep=True)
    psr[arg] = 11
    gsr[arg] = 11

    assert_eq(psr, gsr, check_index_type=True)


def test_series_set_item_index_reference():
    gs1 = cudf.Series([1], index=[7])
    gs2 = cudf.Series([2], index=gs1.index)
    gs1.loc[11] = 2

    ps1 = pd.Series([1], index=[7])
    ps2 = pd.Series([2], index=ps1.index)
    ps1.loc[11] = 2

    assert_eq(ps1, gs1)
    assert_eq(ps2, gs2)
