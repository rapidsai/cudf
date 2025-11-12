# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_220
from cudf.testing import assert_eq
from cudf.testing._utils import expect_warning_if


@pytest.fixture(params=["integer", "signed", "unsigned", "float"])
def downcast(request):
    return request.param


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
        [1, 2**11],
        [1, 2**33],
        [1, 2**63],
        [np.iinfo(np.int64).max, np.iinfo(np.int64).min],
    ],
)
def test_to_numeric_downcast_int(data, downcast):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.to_numeric(ps, downcast=downcast)
    got = cudf.to_numeric(gs, downcast=downcast)

    assert_eq(expected, got)


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0**11],
        [-1.0, -(2.0**11)],
        [1.0, 2.0**33],
        [-1.0, -(2.0**33)],
        [1.0, 2.0**65],
        [-1.0, -(2.0**65)],
        [1.0, float("inf")],
        [1.0, float("-inf")],
        [1.0, float("nan")],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 1.5, 2.6, 3.4],
    ],
)
def test_to_numeric_downcast_float(data, downcast):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.to_numeric(ps, downcast=downcast)
    got = cudf.to_numeric(gs, downcast=downcast)

    assert_eq(expected, got)


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0**129],
        [1.0, 2.0**257],
        [1.0, 1.79e308],
        [-1.0, -(2.0**129)],
        [-1.0, -(2.0**257)],
        [-1.0, -1.79e308],
    ],
)
def test_to_numeric_downcast_large_float(data, downcast):
    if downcast == "float":
        pytest.skip(f"{downcast=} not applicable for test")

    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.to_numeric(ps, downcast=downcast)
    got = cudf.to_numeric(gs, downcast=downcast)

    assert_eq(expected, got)


@pytest.mark.filterwarnings("ignore:overflow encountered in cast")
@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0**129],
        [1.0, 2.0**257],
        [1.0, 1.79e308],
        [-1.0, -(2.0**129)],
        [-1.0, -(2.0**257)],
        [-1.0, -1.79e308],
    ],
)
def test_to_numeric_downcast_large_float_pd_bug(data):
    downcast = "float"
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.to_numeric(ps, downcast=downcast)
    got = cudf.to_numeric(gs, downcast=downcast)

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        ["1", "2", "3"],
        [str(np.iinfo(np.int64).max), str(np.iinfo(np.int64).min)],
    ],
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


@pytest.mark.filterwarnings("ignore:overflow encountered in cast")
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
def test_to_numeric_downcast_string_large_float(data, downcast):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    if downcast == "float":
        expected = pd.to_numeric(ps, downcast=downcast)
        got = cudf.to_numeric(gs, downcast=downcast)

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
        with expect_warning_if(PANDAS_GE_220 and errors == "ignore"):
            expect = pd.to_numeric(data, errors=errors)
        with expect_warning_if(errors == "ignore"):
            got = cudf.to_numeric(data, errors=errors)

        assert_eq(expect, got)


def test_series_to_numeric_bool(downcast):
    data = [True, False, True]
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expect = pd.to_numeric(ps, downcast=downcast)
    got = cudf.to_numeric(gs, downcast=downcast)

    assert_eq(expect, got)


@pytest.mark.parametrize("klass", [cudf.Series, pd.Series])
def test_series_to_numeric_preserve_index_name(klass):
    ser = klass(["1"] * 8, index=range(2, 10), name="name")
    result = cudf.to_numeric(ser)
    expected = cudf.Series([1] * 8, index=range(2, 10), name="name")
    assert_eq(result, expected)
