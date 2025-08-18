# Copyright (c) 2023-2025, NVIDIA CORPORATION.
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "testlist",
    [
        [1, 2, 3, 4],
        [1, 2, 3, 3, 4],
        [10, 9, 8, 7],
        [10, 9, 8, 8, 7],
        ["c", "d", "e", "f"],
        ["c", "d", "e", "e", "f"],
        ["z", "y", "x", "r"],
        ["z", "y", "x", "x", "r"],
    ],
)
def test_series_is_unique_monotonic(testlist):
    series = cudf.Series(testlist)
    series_pd = pd.Series(testlist)

    assert series.is_unique == series_pd.is_unique
    assert series.is_monotonic_increasing == series_pd.is_monotonic_increasing
    assert series.is_monotonic_decreasing == series_pd.is_monotonic_decreasing


@pytest.mark.parametrize(
    "data",
    [
        [pd.Timestamp("2018-01-01"), pd.Timestamp("2019-01-31"), None],
        [1, 2, 3, None],
        [None, 1, 2, 3],
        ["a", "b", "c", None],
        [None, "a", "b", "c"],
    ],
)
def test_is_monotonic_always_false_for_null(data):
    ser = cudf.Series(data)
    assert ser.is_monotonic_increasing is False
    assert ser.is_monotonic_decreasing is False


@pytest.mark.parametrize("box", [cudf.Series, cudf.Index])
@pytest.mark.parametrize(
    "value,na_like",
    [
        [1, None],
        [np.datetime64("2020-01-01", "ns"), np.datetime64("nat", "ns")],
        ["s", None],
        [1.0, np.nan],
    ],
    ids=repr,
)
def test_is_unique(box, value, na_like):
    obj = box([value], nan_as_null=False)
    assert obj.is_unique

    obj = box([value, value], nan_as_null=False)
    assert not obj.is_unique

    obj = box([None, value], nan_as_null=False)
    assert obj.is_unique

    obj = box([None, None, value], nan_as_null=False)
    assert not obj.is_unique

    if na_like is not None:
        obj = box([na_like, value], nan_as_null=False)
        assert obj.is_unique

        obj = box([na_like, na_like], nan_as_null=False)
        assert not obj.is_unique

        try:
            if not np.isnat(na_like):
                # pyarrow coerces nat to null
                obj = box([None, na_like, value], nan_as_null=False)
                assert obj.is_unique
        except TypeError:
            pass


@pytest.fixture(
    params=[
        pd.Series([0, 1, 2, np.nan, 4, None, 6]),
        pd.Series(
            [0, 1, 2, np.nan, 4, None, 6],
            index=["q", "w", "e", "r", "t", "y", "u"],
            name="a",
        ),
        pd.Series([0, 1, 2, 3, 4]),
        pd.Series(["a", "b", "u", "h", "d"]),
        pd.Series([None, None, np.nan, None, np.inf, -np.inf]),
        pd.Series([], dtype="float64"),
        pd.Series(
            [pd.NaT, pd.Timestamp("1939-05-27"), pd.Timestamp("1940-04-25")]
        ),
        pd.Series([np.nan]),
        pd.Series([None]),
        pd.Series(["a", "b", "", "c", None, "e"]),
    ]
)
def ps(request):
    return request.param


def test_series_iter_error():
    gs = cudf.Series([1, 2, 3])

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        iter(gs)

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        gs.items()

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        gs.iteritems()

    with pytest.raises(TypeError):
        iter(gs._column)


@pytest.mark.parametrize("data", [[], [None, None], ["a", None]])
def test_series_size(data):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    assert_eq(psr.size, gsr.size)


def test_set_index_unequal_length():
    s = cudf.Series(dtype="float64")
    with pytest.raises(ValueError):
        s.index = [1, 2, 3]


@pytest.mark.parametrize(
    "data",
    [
        [],
        [1, 2, 3, 4],
        ["a", "b", "c"],
        [1.2, 2.2, 4.5],
        [np.nan, np.nan],
        [None, None, None],
    ],
)
def test_axes(data):
    csr = cudf.Series(data)
    psr = csr.to_pandas()

    expected = psr.axes
    actual = csr.axes

    for e, a in zip(expected, actual, strict=True):
        assert_eq(e, a)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        pytest.param(
            [np.nan, 10, 15, 16],
            marks=pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/49818"
            ),
        ),
        [np.nan, None, 10, 20],
        ["ab", "zx", "pq"],
        ["ab", "zx", None, "pq"],
        [],
    ],
)
def test_series_hasnans(data):
    gs = cudf.Series(data, nan_as_null=False)
    ps = gs.to_pandas(nullable=True)

    # Check type to avoid mixing Python bool and NumPy bool
    assert isinstance(gs.hasnans, bool)
    assert gs.hasnans == ps.hasnans


def test_category_dtype_attribute():
    psr = pd.Series(["a", "b", "a", "c"], dtype="category")
    sr = cudf.Series(["a", "b", "a", "c"], dtype="category")
    assert isinstance(sr.dtype, cudf.CategoricalDtype)
    assert_eq(sr.dtype.categories, psr.dtype.categories)


def test_dtype_dtypes_equal():
    ser = cudf.Series([0])
    assert ser.dtype is ser.dtypes
    assert ser.dtypes is ser.to_pandas().dtypes


@pytest.mark.parametrize("data", [[], [1, 2, 3, 4, 5]])
@pytest.mark.parametrize(
    "scalar",
    [
        1,
        2,
        3,
        "a",
        np.timedelta64(1, "s"),
        np.timedelta64(2, "s"),
        np.timedelta64(2, "D"),
        np.timedelta64(3, "ms"),
        np.timedelta64(4, "us"),
        np.timedelta64(5, "ns"),
        np.timedelta64(6, "ns"),
        np.datetime64(6, "s"),
    ],
)
def test_timedelta_contains(data, timedelta_types_as_str, scalar):
    sr = cudf.Series(data, dtype=timedelta_types_as_str)
    psr = sr.to_pandas()

    expected = scalar in sr
    actual = scalar in psr

    assert_eq(expected, actual)


def test_cai_after_indexing():
    df = cudf.DataFrame({"a": [1, 2, 3]})
    cai1 = df["a"].__cuda_array_interface__
    df[["a"]]
    cai2 = df["a"].__cuda_array_interface__
    assert cai1 == cai2


@pytest.mark.parametrize(
    "data, expected",
    [
        [["2018-01-01", None, "2019-01-31", None, "2018-01-01"], True],
        [
            [
                "2018-01-01",
                "2018-01-02",
                "2019-01-31",
                "2018-03-01",
                "2018-01-01",
            ],
            False,
        ],
        [
            np.array(
                ["2018-01-01", None, "2019-12-30"], dtype="datetime64[ms]"
            ),
            True,
        ],
    ],
)
def test_datetime_has_null_test(data, expected):
    data = cudf.Series(data, dtype="datetime64[ms]")
    pd_data = data.to_pandas()
    count = pd_data.notna().value_counts()
    expected_count = 0
    if False in count.keys():
        expected_count = count[False]

    assert expected is data.has_nulls
    assert expected_count == data.null_count


def test_datetime_has_null_test_pyarrow():
    data = cudf.Series(
        pa.array(
            [0, np.iinfo("int64").min, np.iinfo("int64").max, None],
            type=pa.timestamp("ns"),
        )
    )
    assert data.has_nulls is True
    assert data.null_count == 1


def test_error_values_datetime():
    s = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    with pytest.raises(NotImplementedError, match="cupy does not support"):
        s.values


def test_ndim():
    s = pd.Series(dtype="float64")
    gs = cudf.Series()
    assert s.ndim == gs.ndim
