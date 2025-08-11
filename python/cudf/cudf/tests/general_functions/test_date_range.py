# Copyright (c) 2025, NVIDIA CORPORATION.


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_GE_220,
    PANDAS_VERSION,
)
from cudf.testing import assert_eq
from cudf.testing._utils import (
    expect_warning_if,
)


def test_date_range_freq_default():
    result = pd.date_range("2020-01-01", periods=2, name="foo")
    expected = cudf.date_range("2020-01-01", periods=2, name="foo")
    assert_eq(result, expected)


def test_date_range_tz():
    result = pd.date_range("2020-01-01", periods=2, tz="UTC")
    expected = cudf.date_range("2020-01-01", periods=2, tz="UTC")
    assert_eq(result, expected)

    result = pd.date_range("2020-01-01", "2020-01-02", periods=2, tz="UTC")
    expected = cudf.date_range("2020-01-01", "2020-01-02", periods=2, tz="UTC")
    assert_eq(result, expected)


def test_date_range_start_end_divisible_by_freq():
    result = cudf.date_range("2011-01-01", "2011-01-02", freq="h")
    expected = pd.date_range("2011-01-01", "2011-01-02", freq="h")
    assert_eq(result, expected)


@pytest.fixture(
    params=[
        "2000-02-13 08:41:06",
        "1996-11-21 04:05:30",
        "1970-01-01 00:00:00",
        "1831-05-08 15:23:21",
    ],
    ids=["leap_year", "non_leap_year", "unix_epoch_time_0", "random_date"],
)
def start(request):
    return request.param


@pytest.fixture(
    params=[
        "2000-02-13 08:41:06",
        "1996-11-21 04:05:30",
        "1970-01-01 00:00:00",
        "1831-05-08 15:23:21",
    ],
    ids=["leap_year", "non_leap_year", "unix_epoch_time_0", "random_date"],
)
def end(request):
    return request.param


@pytest.fixture(params=[1, 10])
def periods(request):
    return request.param


@pytest.fixture(
    params=[
        {"months": 3, "years": 1},
        {"hours": 10, "days": 57, "nanoseconds": 3},
        "83D",
        "17h",
        "-680min",
        "110546s",
        "110546789ms",
        "110546789248us",
    ]
)
def freq(request):
    return request.param


def test_date_range_start_end_periods(start, end, periods):
    expect = pd.date_range(start=start, end=end, periods=periods, name="a")
    got = cudf.date_range(start=start, end=end, periods=periods, name="a")

    np.testing.assert_allclose(
        expect.to_numpy().astype("int64"),
        got.to_pandas().to_numpy().astype("int64"),
    )


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="https://github.com/pandas-dev/pandas/issues/46877",
)
def test_date_range_end_freq_periods(end, freq, periods):
    if isinstance(freq, str):
        _gfreq = _pfreq = freq
    else:
        _gfreq = cudf.DateOffset(**freq)
        _pfreq = pd.DateOffset(**freq)

    expect = pd.date_range(end=end, periods=periods, freq=_pfreq, name="a")
    got = cudf.date_range(end=end, periods=periods, freq=_gfreq, name="a")

    np.testing.assert_allclose(
        expect.to_numpy().astype("int64"),
        got.to_pandas().to_numpy().astype("int64"),
    )


def test_date_range_freq_does_not_divide_range():
    expect = pd.date_range(
        "2001-01-01 00:00:00.000000", "2001-01-01 00:00:00.000010", freq="3us"
    )
    got = cudf.date_range(
        "2001-01-01 00:00:00.000000", "2001-01-01 00:00:00.000010", freq="3us"
    )
    np.testing.assert_allclose(
        expect.to_numpy().astype("int64"),
        got.to_pandas().to_numpy().astype("int64"),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nanoseconds": 1},
        {"months": 1},
    ],
)
def test_date_range_raise_overflow(kwargs):
    start = np.datetime64(np.iinfo("int64").max, "ns")
    periods = 2
    freq = cudf.DateOffset(**kwargs)
    with pytest.raises(pd.errors.OutOfBoundsDatetime):
        cudf.date_range(start=start, periods=periods, freq=freq)


@pytest.mark.parametrize(
    "freqstr_unsupported",
    [
        "1ME",
        "2SME",
        "3MS",
        "4BME",
        "5CBME",
        "6SMS",
        "7BMS",
        "8CBMS",
        "QE",
        "2BQE",
        "3BQS",
        "10YE",
        "9BYE",
        "8YS",
        "7BYS",
        "bh",
        "B",
    ],
)
def test_date_range_raise_unsupported(freqstr_unsupported):
    if not PANDAS_GE_220 and freqstr_unsupported.endswith("E"):
        pytest.skip(reason="YE, etc. support was added in pandas 2.2")

    s, e = "2001-01-01", "2008-01-31"
    pd.date_range(start=s, end=e, freq=freqstr_unsupported)
    with pytest.raises(ValueError, match="does not yet support"):
        cudf.date_range(start=s, end=e, freq=freqstr_unsupported)

    # We also check that these values are unsupported when using lowercase
    # characters. We exclude the value 3MS (every 3 month starts) because 3ms
    # is a valid frequency for every 3 milliseconds.
    if freqstr_unsupported != "3MS":
        freqstr_unsupported = freqstr_unsupported.lower()
        with pytest.raises(ValueError, match="does not yet support"):
            with expect_warning_if(
                PANDAS_GE_220 and freqstr_unsupported not in {"b", "bh"}
            ):
                cudf.date_range(start=s, end=e, freq=freqstr_unsupported)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_date_range_start_freq_periods(start, freq, periods):
    if isinstance(freq, str):
        _gfreq = _pfreq = freq
    else:
        _gfreq = cudf.DateOffset(**freq)
        _pfreq = pd.DateOffset(**freq)

    expect = pd.date_range(start=start, periods=periods, freq=_pfreq, name="a")
    got = cudf.date_range(start=start, periods=periods, freq=_gfreq, name="a")

    np.testing.assert_allclose(
        expect.to_numpy().astype("int64"),
        got.to_pandas().to_numpy().astype("int64"),
    )


def test_daterange_pandas_compatibility():
    with cudf.option_context("mode.pandas_compatible", True):
        expected = pd.date_range(
            "2010-01-01", "2010-02-01", periods=10, name="times"
        )
        actual = cudf.date_range(
            "2010-01-01", "2010-02-01", periods=10, name="times"
        )
    assert_eq(expected, actual)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_date_range_start_end_freq(start, end, freq):
    if isinstance(freq, str):
        _gfreq = _pfreq = freq
    else:
        _gfreq = cudf.DateOffset(**freq)
        _pfreq = pd.DateOffset(**freq)

    expect = pd.date_range(start=start, end=end, freq=_pfreq, name="a")
    got = cudf.date_range(start=start, end=end, freq=_gfreq, name="a")

    np.testing.assert_allclose(
        expect.to_numpy().astype("int64"),
        got.to_pandas().to_numpy().astype("int64"),
    )
