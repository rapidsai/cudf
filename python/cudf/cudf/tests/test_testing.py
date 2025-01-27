# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core.column.column import as_column
from cudf.testing import (
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)
from cudf.testing._utils import (
    NUMERIC_TYPES,
    OTHER_TYPES,
    assert_column_memory_eq,
    assert_column_memory_ne,
)
from cudf.testing.testing import assert_column_equal, assert_eq


@pytest.fixture(
    params=[
        pa.array([*range(10)]),
        pa.array(["hello", "world", "rapids", "AI"]),
        pa.array([[1, 2, 3], [4, 5], [6], [], [7]]),
        pa.array([{"f0": "hello", "f1": 42}, {"f0": "world", "f1": 3}]),
    ]
)
def arrow_arrays(request):
    return request.param


@pytest.mark.parametrize("rdata", [[1, 2, 5], [1, 2, 6], [1, 2, 5, 6]])
@pytest.mark.parametrize("exact", ["equiv", True, False])
@pytest.mark.parametrize("check_names", [True, False])
@pytest.mark.parametrize("rname", ["a", "b"])
@pytest.mark.parametrize("check_categorical", [True, False])
@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + OTHER_TYPES + ["datetime64[ns]"]
)
def test_basic_assert_index_equal(
    rdata,
    exact,
    check_names,
    rname,
    check_categorical,
    dtype,
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
        if (kind is TypeError) and (
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
    rdata,
    rname,
    check_names,
    check_category_order,
    check_categorical,
    dtype,
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
    "other",
    [
        as_column(["1", "2", "3"]),
        as_column([[1], [2], [3]]),
        as_column([{"a": 1}, {"a": 2}, {"a": 3}]),
    ],
)
def test_assert_column_equal_dtype_edge_cases(other):
    # string series should be 100% different
    # even when the elements are the same
    base = as_column([1, 2, 3])

    # for these dtypes, the diff should always be 100% regardless of the values
    with pytest.raises(
        AssertionError, match=r".*values are different \(100.0 %\).*"
    ):
        assert_column_equal(base, other, check_dtype=False)

    # the exceptions are the empty and all null cases
    assert_column_equal(base.slice(0, 0), other.slice(0, 0), check_dtype=False)
    assert_column_equal(other.slice(0, 0), base.slice(0, 0), check_dtype=False)

    base = as_column(cudf.NA, length=len(base), dtype=base.dtype)
    other = as_column(cudf.NA, length=len(other), dtype=other.dtype)

    assert_column_equal(base, other, check_dtype=False)
    assert_column_equal(other, base, check_dtype=False)


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
    "dtype",
    ["int8", "int16", "int32", "int64"],
)
@pytest.mark.parametrize("exact", ["equiv", True, False])
def test_range_index_and_int_index_eqaulity(dtype, exact):
    pidx1 = pd.RangeIndex(0, stop=5, step=1)
    pidx2 = pd.Index([0, 1, 2, 3, 4])
    idx1 = cudf.from_pandas(pidx1)
    idx2 = cudf.Index([0, 1, 2, 3, 4], dtype=dtype)

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


def test_assert_column_memory_basic(arrow_arrays):
    left = cudf.core.column.ColumnBase.from_arrow(arrow_arrays)
    right = cudf.core.column.ColumnBase.from_arrow(arrow_arrays)

    with pytest.raises(AssertionError):
        assert_column_memory_eq(left, right)
    assert_column_memory_ne(left, right)


def test_assert_column_memory_slice(arrow_arrays):
    col = cudf.core.column.ColumnBase.from_arrow(arrow_arrays)
    left = col.slice(0, 1)
    right = col.slice(1, 2)

    with pytest.raises(AssertionError):
        assert_column_memory_eq(left, right)
    assert_column_memory_ne(left, right)

    with pytest.raises(AssertionError):
        assert_column_memory_eq(left, col)
    assert_column_memory_ne(left, col)

    with pytest.raises(AssertionError):
        assert_column_memory_eq(right, col)
    assert_column_memory_ne(right, col)


def test_assert_column_memory_basic_same(arrow_arrays):
    data = cudf.core.column.ColumnBase.from_arrow(arrow_arrays)
    buf = cudf.core.buffer.as_buffer(data.base_data)

    left = cudf.core.column.build_column(buf, dtype=np.int8)
    right = cudf.core.column.build_column(buf, dtype=np.int8)

    assert_column_memory_eq(left, right)
    with pytest.raises(AssertionError):
        assert_column_memory_ne(left, right)
