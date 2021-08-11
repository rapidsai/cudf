# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import (
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)
from cudf.testing._utils import NUMERIC_TYPES, OTHER_TYPES, assert_eq


@pytest.mark.parametrize("rdata", [[1, 2, 5], [1, 2, 6], [1, 2, 5, 6]])
@pytest.mark.parametrize("exact", ["equiv", True, False])
@pytest.mark.parametrize("check_names", [True, False])
@pytest.mark.parametrize("rname", ["a", "b"])
@pytest.mark.parametrize("check_categorical", [True, False])
@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + OTHER_TYPES + ["datetime64[ns]"]
)
def test_basic_assert_index_equal(
    rdata, exact, check_names, rname, check_categorical, dtype,
):
    p_left = pd.Index([1, 2, 3], name="a", dtype=dtype)
    p_right = pd.Index(rdata, name=rname, dtype=dtype)

    left = cudf.from_pandas(p_left)
    right = cudf.from_pandas(p_right)

    kind = None
    try:
        pd.testing.assert_index_equal(
            p_left,
            p_right,
            exact=exact,
            check_names=check_names,
            check_categorical=check_categorical,
        )
    except BaseException as e:
        kind = type(e)
        msg = str(e)

    if kind is not None:
        if (kind == TypeError) and (
            msg
            == (
                "Categoricals can only be compared "
                "if 'categories' are the same."
            )
        ):
            kind = AssertionError
        with pytest.raises(kind):
            assert_index_equal(
                left,
                right,
                exact=exact,
                check_names=check_names,
                check_categorical=check_categorical,
            )
    else:
        assert_index_equal(
            left,
            right,
            exact=exact,
            check_names=check_names,
            check_categorical=check_categorical,
        )


@pytest.mark.parametrize("rdata", [[1, 2, 5], [1, 2, 6], [1, 2, 5, 6]])
@pytest.mark.parametrize("check_names", [True, False])
@pytest.mark.parametrize("rname", ["a", "b"])
@pytest.mark.parametrize("check_category_order", [True, False])
@pytest.mark.parametrize("check_categorical", [True, False])
@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + OTHER_TYPES + ["datetime64[ns]"]
)
def test_basic_assert_series_equal(
    rdata, rname, check_names, check_category_order, check_categorical, dtype,
):

    p_left = pd.Series([1, 2, 3], name="a", dtype=dtype)
    p_right = pd.Series(rdata, name=rname, dtype=dtype)

    left = cudf.from_pandas(p_left)
    right = cudf.from_pandas(p_right)

    kind = None
    try:
        pd.testing.assert_series_equal(
            p_left,
            p_right,
            check_names=check_names,
            check_categorical=check_categorical,
            check_category_order=check_category_order,
        )
    except BaseException as e:
        kind = type(e)

    if kind is not None:
        with pytest.raises(kind):
            assert_series_equal(
                left,
                right,
                check_names=check_names,
                check_categorical=check_categorical,
                check_category_order=check_category_order,
            )
    else:
        assert_series_equal(
            left,
            right,
            check_names=check_names,
            check_categorical=check_categorical,
            check_category_order=check_category_order,
        )


@pytest.mark.parametrize(
    "rdtype", [["int8", "int16", "int64"], ["int64", "int16", "int8"]]
)
@pytest.mark.parametrize("rname", [["a", "b", "c"], ["b", "c", "a"]])
@pytest.mark.parametrize("index", [[1, 2, 3], [3, 2, 1]])
@pytest.mark.parametrize("check_exact", [True, False])
@pytest.mark.parametrize("check_dtype", [True, False])
@pytest.mark.parametrize("check_names", [True, False])
@pytest.mark.parametrize("check_like", [True, False])
@pytest.mark.parametrize("mismatch", [True, False])
def test_basic_assert_frame_equal(
    rdtype,
    rname,
    index,
    check_exact,
    check_dtype,
    check_names,
    check_like,
    mismatch,
):
    data = [1, 2, 1]
    p_left = pd.DataFrame(index=[1, 2, 3])
    p_left["a"] = np.array(data, dtype="int8")
    p_left["b"] = np.array(data, dtype="int16")
    if mismatch:
        p_left["c"] = np.array([1, 2, 3], dtype="int64")
    else:
        p_left["c"] = np.array(data, dtype="int64")

    p_right = pd.DataFrame(index=index)
    for dtype, name in zip(rdtype, rname):
        p_right[name] = np.array(data, dtype=dtype)

    left = cudf.from_pandas(p_left)
    right = cudf.from_pandas(p_right)

    kind = None
    try:
        pd.testing.assert_frame_equal(
            p_left,
            p_right,
            check_exact=check_exact,
            check_dtype=check_dtype,
            check_names=check_names,
            check_like=check_like,
        )
    except BaseException as e:
        kind = type(e)

    if kind is not None:
        with pytest.raises(kind):
            assert_frame_equal(
                left,
                right,
                check_exact=check_exact,
                check_dtype=check_dtype,
                check_names=check_names,
                check_like=check_like,
            )
    else:
        assert_frame_equal(
            left,
            right,
            check_exact=check_exact,
            check_dtype=check_dtype,
            check_names=check_names,
            check_like=check_like,
        )


@pytest.mark.parametrize("rdata", [[0, 1, 2, 3], [0, 1, 2, 4]])
@pytest.mark.parametrize("check_datetimelike_compat", [True, False])
def test_datetime_like_compaibility(rdata, check_datetimelike_compat):
    psr1 = pd.Series([0, 1, 2, 3], dtype="datetime64[ns]")
    psr2 = pd.Series(rdata, dtype="datetime64[ns]").astype("str")

    sr1 = cudf.from_pandas(psr1)
    sr2 = cudf.from_pandas(psr2)

    kind = None
    try:
        pd.testing.assert_series_equal(
            psr1, psr2, check_datetimelike_compat=check_datetimelike_compat
        )
    except BaseException as e:
        kind = type(e)

    if kind is not None:
        with pytest.raises(kind):
            assert_series_equal(
                sr1, sr2, check_datetimelike_compat=check_datetimelike_compat
            )
    else:
        assert_series_equal(
            sr1, sr2, check_datetimelike_compat=check_datetimelike_compat
        )


@pytest.mark.parametrize(
    "rdata",
    [
        [[0, 1, 2, 3], ["G", "O", "N", "E"]],
        [[0, 1, 2, 4], ["G", "O", "N", "E"]],
    ],
)
def test_multiindex_equal(rdata):
    pidx1 = pd.MultiIndex.from_arrays(
        [[0, 1, 2, 3], ["G", "O", "N", "E"]], names=("n", "id")
    )
    pidx2 = pd.MultiIndex.from_arrays(rdata, names=("n", "id"))

    idx1 = cudf.from_pandas(pidx1)
    idx2 = cudf.from_pandas(pidx2)

    kind = None
    try:
        pd.testing.assert_index_equal(pidx1, pidx2)
    except BaseException as e:
        kind = type(e)

    if kind is not None:
        with pytest.raises(kind):
            assert_index_equal(idx1, idx2)
    else:
        assert_index_equal(idx1, idx2)


@pytest.mark.parametrize("dtype", ["int8", "uint8", "float32"])
@pytest.mark.parametrize("check_exact", [True, False])
@pytest.mark.parametrize("check_dtype", [True, False])
def test_series_different_type_cases(dtype, check_exact, check_dtype):
    data = [0, 1, 2, 3]

    psr1 = pd.Series(data, dtype="uint8")
    psr2 = pd.Series(data, dtype=dtype)

    sr1 = cudf.from_pandas(psr1)
    sr2 = cudf.from_pandas(psr2)

    kind = None
    try:
        pd.testing.assert_series_equal(
            psr1, psr2, check_exact=check_exact, check_dtype=check_dtype
        )
    except BaseException as e:
        kind = type(e)

    if kind is not None:
        with pytest.raises(kind):
            assert_series_equal(
                sr1, sr2, check_exact=check_exact, check_dtype=check_dtype
            )
    else:
        assert_series_equal(
            sr1, sr2, check_exact=check_exact, check_dtype=check_dtype
        )


@pytest.mark.parametrize(
    "index",
    [cudf.Int8Index, cudf.Int16Index, cudf.Int32Index, cudf.Int64Index],
)
@pytest.mark.parametrize("exact", ["equiv", True, False])
def test_range_index_and_int_index_eqaulity(index, exact):
    pidx1 = pd.RangeIndex(0, stop=5, step=1)
    pidx2 = pd.Index([0, 1, 2, 3, 4])
    idx1 = cudf.from_pandas(pidx1)
    idx2 = index([0, 1, 2, 3, 4])

    kind = None
    try:
        pd.testing.assert_index_equal(pidx1, pidx2, exact=exact)
    except BaseException as e:
        kind = type(e)

    if kind is not None:
        with pytest.raises(kind):
            assert_index_equal(idx1, idx2, exact=exact)
    else:
        assert_index_equal(idx1, idx2, exact=exact)


@pytest.mark.parametrize(
    "left, right",
    [
        (1493282, 1493282),
        (1493282.0, 1493282.0 + 1e-8),
        ("abc", "abc"),
        (0, np.array(0)),
        (
            np.datetime64(123456, "ns"),
            pd.Timestamp(np.datetime64(123456, "ns")),
        ),
        ("int64", np.dtype("int64")),
        (np.nan, np.nan),
    ],
)
def test_basic_scalar_equality(left, right):
    assert_eq(left, right)


@pytest.mark.parametrize(
    "left, right",
    [
        (1493282, 1493274),
        (1493282.0, 1493282.0 + 1e-6),
        ("abc", "abd"),
        (0, np.array(1)),
        (
            np.datetime64(123456, "ns"),
            pd.Timestamp(np.datetime64(123457, "ns")),
        ),
        ("int64", np.dtype("int32")),
    ],
)
def test_basic_scalar_inequality(left, right):
    with pytest.raises(AssertionError, match=r".*not (almost )?equal.*"):
        assert_eq(left, right)
