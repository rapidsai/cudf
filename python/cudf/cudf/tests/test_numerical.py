import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import Series
from cudf.tests.utils import assert_eq


def test_can_cast_safely_same_kind():
    # 'i' -> 'i'
    data = Series([1, 2, 3], dtype="int32")._column
    to_dtype = np.dtype("int64")

    assert data.can_cast_safely(to_dtype)

    data = Series([1, 2, 3], dtype="int64")._column
    to_dtype = np.dtype("int32")

    assert data.can_cast_safely(to_dtype)

    data = Series([1, 2, 2 ** 31], dtype="int64")._column
    assert not data.can_cast_safely(to_dtype)

    # 'u' -> 'u'
    data = Series([1, 2, 3], dtype="uint32")._column
    to_dtype = np.dtype("uint64")

    assert data.can_cast_safely(to_dtype)

    data = Series([1, 2, 3], dtype="uint64")._column
    to_dtype = np.dtype("uint32")

    assert data.can_cast_safely(to_dtype)

    data = Series([1, 2, 2 ** 33], dtype="uint64")._column
    assert not data.can_cast_safely(to_dtype)

    # 'f' -> 'f'
    data = Series([np.inf, 1.0], dtype="float64")._column
    to_dtype = np.dtype("float32")
    assert data.can_cast_safely(to_dtype)

    data = Series([np.finfo("float32").max * 2, 1.0], dtype="float64")._column
    to_dtype = np.dtype("float32")
    assert not data.can_cast_safely(to_dtype)


def test_can_cast_safely_mixed_kind():
    data = Series([1, 2, 3], dtype="int32")._column
    to_dtype = np.dtype("float32")
    assert data.can_cast_safely(to_dtype)

    # too big to fit into f32 exactly
    data = Series([1, 2, 2 ** 24 + 1], dtype="int32")._column
    assert not data.can_cast_safely(to_dtype)

    data = Series([1, 2, 3], dtype="uint32")._column
    to_dtype = np.dtype("float32")
    assert data.can_cast_safely(to_dtype)

    # too big to fit into f32 exactly
    data = Series([1, 2, 2 ** 24 + 1], dtype="uint32")._column
    assert not data.can_cast_safely(to_dtype)

    to_dtype = np.dtype("float64")
    assert data.can_cast_safely(to_dtype)

    data = Series([1.0, 2.0, 3.0], dtype="float32")._column
    to_dtype = np.dtype("int32")
    assert data.can_cast_safely(to_dtype)

    # not integer float
    data = Series([1.0, 2.0, 3.5], dtype="float32")._column
    assert not data.can_cast_safely(to_dtype)

    data = Series([10.0, 11.0, 2000.0], dtype="float64")._column
    assert data.can_cast_safely(to_dtype)

    # float out of int range
    data = Series([1.0, 2.0, 1.0 * (2 ** 31)], dtype="float32")._column
    assert not data.can_cast_safely(to_dtype)

    # negative signed integers casting to unsigned integers
    data = Series([-1, 0, 1], dtype="int32")._column
    to_dtype = np.dtype("uint32")
    assert not data.can_cast_safely(to_dtype)


@pytest.mark.xfail(
    reason="cuDF null <-> pd.NA compatibility not yet supported"
)
def test_to_pandas_nullable_integer():
    gsr_not_null = Series([1, 2, 3])
    gsr_has_null = Series([1, 2, None])

    psr_not_null = pd.Series([1, 2, 3], dtype="int64")
    psr_has_null = pd.Series([1, 2, None], dtype="Int64")

    assert_eq(gsr_not_null.to_pandas(), psr_not_null)
    assert_eq(gsr_has_null.to_pandas(), psr_has_null)


@pytest.mark.xfail(
    reason="cuDF null <-> pd.NA compatibility not yet supported"
)
def test_to_pandas_nullable_bool():
    gsr_not_null = Series([True, False, True])
    gsr_has_null = Series([True, False, None])

    psr_not_null = pd.Series([True, False, True], dtype="bool")
    psr_has_null = pd.Series([True, False, None], dtype="boolean")

    assert_eq(gsr_not_null.to_pandas(), psr_not_null)
    assert_eq(gsr_has_null.to_pandas(), psr_has_null)


def test_can_cast_safely_has_nulls():
    data = Series([1, 2, 3, None], dtype="float32")._column
    to_dtype = np.dtype("int64")

    assert data.can_cast_safely(to_dtype)

    data = Series([1, 2, 3.1, None], dtype="float32")._column
    assert not data.can_cast_safely(to_dtype)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        (1.0, 2.0, 3.0),
        [float("nan"), None],
        np.array([1, 2.0, -3, float("nan")]),
        pd.Series(["123", "2.0"]),
        pd.Series(["1.0", "2.", "-.3", "1e6"]),
        pd.Series(
            ["1", "2", "3"],
            dtype=pd.CategoricalDtype(categories=["1", "2", "3"]),
        ),
        pd.Series(
            ["1.0", "2.0", "3.0"],
            dtype=pd.CategoricalDtype(categories=["1.0", "2.0", "3.0"]),
        ),
        # Categories with nulls
        pd.Series([1, 2, 3], dtype=pd.CategoricalDtype(categories=[1, 2])),
        pd.Series(
            [5.0, 6.0], dtype=pd.CategoricalDtype(categories=[5.0, 6.0])
        ),
        pd.Series(
            ["2020-08-01 08:00:00", "1960-08-01 08:00:00"],
            dtype=np.dtype("<M8[ns]"),
        ),
        pd.Series(
            [pd.Timedelta(days=1, seconds=1), pd.Timedelta("-3 seconds 4ms")],
            dtype=np.dtype("<m8[ns]"),
        ),
        [
            "inf",
            "-inf",
            "+inf",
            "infinity",
            "-infinity",
            "+infinity",
            "inFInity",
        ],
    ],
)
def test_to_numeric_basic_1d(data):
    expected = pd.to_numeric(data)
    got = cudf.to_numeric(data)

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2 ** 11],
        [1, 2 ** 33],
        [1, 2 ** 63],
        [np.iinfo(np.int64).max, np.iinfo(np.int64).min],
    ],
)
@pytest.mark.parametrize(
    "downcast", ["integer", "signed", "unsigned", "float"]
)
def test_to_numeric_downcast_int(data, downcast):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.to_numeric(ps, downcast=downcast)
    got = cudf.to_numeric(gs, downcast=downcast)

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0 ** 11],
        [-1.0, -(2.0 ** 11)],
        [1.0, 2.0 ** 33],
        [-1.0, -(2.0 ** 33)],
        [1.0, 2.0 ** 65],
        [-1.0, -(2.0 ** 65)],
        [1.0, float("inf")],
        [1.0, float("-inf")],
        [1.0, float("nan")],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 1.5, 2.6, 3.4],
    ],
)
@pytest.mark.parametrize(
    "downcast", ["signed", "integer", "unsigned", "float"]
)
def test_to_numeric_downcast_float(data, downcast):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.to_numeric(ps, downcast=downcast)
    got = cudf.to_numeric(gs, downcast=downcast)

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0 ** 129],
        [1.0, 2.0 ** 257],
        [1.0, 1.79e308],
        [-1.0, -(2.0 ** 129)],
        [-1.0, -(2.0 ** 257)],
        [-1.0, -1.79e308],
    ],
)
@pytest.mark.parametrize("downcast", ["signed", "integer", "unsigned"])
def test_to_numeric_downcast_large_float(data, downcast):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.to_numeric(ps, downcast=downcast)
    got = cudf.to_numeric(gs, downcast=downcast)

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0 ** 129],
        [1.0, 2.0 ** 257],
        [1.0, 1.79e308],
        [-1.0, -(2.0 ** 129)],
        [-1.0, -(2.0 ** 257)],
        [-1.0, -1.79e308],
    ],
)
@pytest.mark.parametrize("downcast", ["float"])
def test_to_numeric_downcast_large_float_pd_bug(data, downcast):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.to_numeric(ps, downcast=downcast)
    got = cudf.to_numeric(gs, downcast=downcast)

    # Pandas bug: https://github.com/pandas-dev/pandas/issues/19729
    with pytest.raises(AssertionError, match="Series are different"):
        assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        ["1", "2", "3"],
        [str(np.iinfo(np.int64).max), str(np.iinfo(np.int64).min)],
    ],
)
@pytest.mark.parametrize(
    "downcast", ["signed", "integer", "unsigned", "float"]
)
def test_to_numeric_downcast_string_int(data, downcast):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.to_numeric(ps, downcast=downcast)
    got = cudf.to_numeric(gs, downcast=downcast)

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        [""],  # pure empty strings
        ["10.0", "11.0", "2e3"],
        ["1.0", "2e3"],
        ["1", "10", "1.0", "2e3"],  # int-float mixed
        ["1", "10", "1.0", "2e3", "2e+3", "2e-3"],
        ["1", "10", "1.0", "2e3", "", ""],  # mixed empty strings
    ],
)
@pytest.mark.parametrize(
    "downcast", ["signed", "integer", "unsigned", "float"]
)
def test_to_numeric_downcast_string_float(data, downcast):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.to_numeric(ps, downcast=downcast)

    if downcast in {"signed", "integer", "unsigned"}:
        with pytest.warns(
            UserWarning,
            match="Downcasting from float to int "
            "will be limited by float32 precision.",
        ):
            got = cudf.to_numeric(gs, downcast=downcast)
    else:
        got = cudf.to_numeric(gs, downcast=downcast)

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        ["2e128", "-2e128"],
        [
            "1.79769313486231e308",
            "-1.79769313486231e308",
        ],  # 2 digits relaxed from np.finfo(np.float64).min/max
    ],
)
@pytest.mark.parametrize(
    "downcast", ["signed", "integer", "unsigned", "float"]
)
def test_to_numeric_downcast_string_large_float(data, downcast):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    if downcast == "float":
        expected = pd.to_numeric(ps, downcast=downcast)
        got = cudf.to_numeric(gs, downcast=downcast)

        # Pandas bug: https://github.com/pandas-dev/pandas/issues/19729
        with pytest.raises(AssertionError, match="Series are different"):
            assert_eq(expected, got)
    else:
        expected = pd.Series([np.inf, -np.inf])
        with pytest.warns(
            UserWarning,
            match="Downcasting from float to int "
            "will be limited by float32 precision.",
        ):
            got = cudf.to_numeric(gs, downcast=downcast)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        pd.Series(["1", "a", "3"]),
        pd.Series(["1", "a", "3", ""]),  # mix of unconvertible and empty str
    ],
)
@pytest.mark.parametrize("errors", ["ignore", "raise", "coerce"])
def test_to_numeric_error(data, errors):
    if errors == "raise":
        with pytest.raises(
            ValueError, match="Unable to convert some strings to numerics."
        ):
            cudf.to_numeric(data, errors=errors)
    else:
        expect = pd.to_numeric(data, errors=errors)
        got = cudf.to_numeric(data, errors=errors)

        assert_eq(expect, got)
