# Copyright (c) 2023, NVIDIA CORPORATION.
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_210
from cudf.core.index import IntervalIndex, interval_range
from cudf.testing._utils import assert_eq


def test_interval_constructor_default_closed():
    idx = cudf.IntervalIndex([pd.Interval(0, 1)])
    assert idx.closed == "right"
    assert idx.dtype.closed == "right"


def test_interval_to_arrow():
    expect = pa.Array.from_pandas(pd.IntervalIndex([pd.Interval(0, 1)]))
    got = cudf.IntervalIndex([pd.Interval(0, 1)]).to_arrow()
    assert_eq(expect, got)


INTERVAL_BOUNDARY_TYPES = [
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    cudf.Scalar,
]


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("start", [0, 1, 2, 3])
@pytest.mark.parametrize("end", [4, 5, 6, 7])
def test_interval_range_basic(start, end, closed):
    pindex = pd.interval_range(start=start, end=end, closed=closed)
    gindex = cudf.interval_range(start=start, end=end, closed=closed)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("start_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("end_t", INTERVAL_BOUNDARY_TYPES)
def test_interval_range_dtype_basic(start_t, end_t):
    start, end = start_t(24), end_t(42)
    start_val = start.value if isinstance(start, cudf.Scalar) else start
    end_val = end.value if isinstance(end, cudf.Scalar) else end
    pindex = pd.interval_range(start=start_val, end=end_val, closed="left")
    gindex = cudf.interval_range(start=start, end=end, closed="left")

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("start", [0])
@pytest.mark.parametrize("end", [0])
def test_interval_range_empty(start, end, closed):
    pindex = pd.interval_range(start=start, end=end, closed=closed)
    gindex = cudf.interval_range(start=start, end=end, closed=closed)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("freq", [1, 2, 3])
@pytest.mark.parametrize("start", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("end", [6, 8, 10, 43, 70])
def test_interval_range_freq_basic(start, end, freq, closed):
    pindex = pd.interval_range(start=start, end=end, freq=freq, closed=closed)
    gindex = cudf.interval_range(
        start=start, end=end, freq=freq, closed=closed
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("start_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("end_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("freq_t", INTERVAL_BOUNDARY_TYPES)
def test_interval_range_freq_basic_dtype(start_t, end_t, freq_t):
    start, end, freq = start_t(5), end_t(70), freq_t(3)
    start_val = start.value if isinstance(start, cudf.Scalar) else start
    end_val = end.value if isinstance(end, cudf.Scalar) else end
    freq_val = freq.value if isinstance(freq, cudf.Scalar) else freq
    pindex = pd.interval_range(
        start=start_val, end=end_val, freq=freq_val, closed="left"
    )
    gindex = cudf.interval_range(
        start=start, end=end, freq=freq, closed="left"
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("periods", [1, 1.0, 2, 2.0, 3.0, 3])
@pytest.mark.parametrize("start", [0, 0.0, 1.0, 1, 2, 2.0, 3.0, 3])
@pytest.mark.parametrize("end", [4, 4.0, 5.0, 5, 6, 6.0, 7.0, 7])
def test_interval_range_periods_basic(start, end, periods, closed):
    pindex = pd.interval_range(
        start=start, end=end, periods=periods, closed=closed
    )
    gindex = cudf.interval_range(
        start=start, end=end, periods=periods, closed=closed
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("start_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("end_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("periods_t", INTERVAL_BOUNDARY_TYPES)
def test_interval_range_periods_basic_dtype(start_t, end_t, periods_t):
    start, end, periods = start_t(0), end_t(4), periods_t(1.0)
    start_val = start.value if isinstance(start, cudf.Scalar) else start
    end_val = end.value if isinstance(end, cudf.Scalar) else end
    periods_val = (
        periods.value if isinstance(periods, cudf.Scalar) else periods
    )
    pindex = pd.interval_range(
        start=start_val, end=end_val, periods=periods_val, closed="left"
    )
    gindex = cudf.interval_range(
        start=start, end=end, periods=periods, closed="left"
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("periods", [1, 2, 3])
@pytest.mark.parametrize("freq", [1, 2, 3, 4])
@pytest.mark.parametrize("end", [4, 8, 9, 10])
def test_interval_range_periods_freq_end(end, freq, periods, closed):
    pindex = pd.interval_range(
        end=end, freq=freq, periods=periods, closed=closed
    )
    gindex = cudf.interval_range(
        end=end, freq=freq, periods=periods, closed=closed
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("periods_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("freq_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("end_t", INTERVAL_BOUNDARY_TYPES)
def test_interval_range_periods_freq_end_dtype(periods_t, freq_t, end_t):
    periods, freq, end = periods_t(2), freq_t(3), end_t(10)
    freq_val = freq.value if isinstance(freq, cudf.Scalar) else freq
    end_val = end.value if isinstance(end, cudf.Scalar) else end
    periods_val = (
        periods.value if isinstance(periods, cudf.Scalar) else periods
    )
    pindex = pd.interval_range(
        end=end_val, freq=freq_val, periods=periods_val, closed="left"
    )
    gindex = cudf.interval_range(
        end=end, freq=freq, periods=periods, closed="left"
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("periods", [1, 2, 3])
@pytest.mark.parametrize("freq", [1, 2, 3, 4])
@pytest.mark.parametrize("start", [1, 4, 9, 12])
def test_interval_range_periods_freq_start(start, freq, periods, closed):
    pindex = pd.interval_range(
        start=start, freq=freq, periods=periods, closed=closed
    )
    gindex = cudf.interval_range(
        start=start, freq=freq, periods=periods, closed=closed
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("periods_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("freq_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("start_t", INTERVAL_BOUNDARY_TYPES)
def test_interval_range_periods_freq_start_dtype(periods_t, freq_t, start_t):
    periods, freq, start = periods_t(2), freq_t(3), start_t(9)
    freq_val = freq.value if isinstance(freq, cudf.Scalar) else freq
    start_val = start.value if isinstance(start, cudf.Scalar) else start
    periods_val = (
        periods.value if isinstance(periods, cudf.Scalar) else periods
    )
    pindex = pd.interval_range(
        start=start_val, freq=freq_val, periods=periods_val, closed="left"
    )
    gindex = cudf.interval_range(
        start=start, freq=freq, periods=periods, closed="left"
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["right", "left", "both", "neither"])
@pytest.mark.parametrize(
    "data",
    [
        ([pd.Interval(30, 50)]),
        ([pd.Interval(0, 3), pd.Interval(1, 7)]),
        ([pd.Interval(0.2, 60.3), pd.Interval(1, 7), pd.Interval(0, 0)]),
        ([]),
    ],
)
def test_interval_index_basic(data, closed):
    pindex = pd.IntervalIndex(data, closed=closed)
    gindex = IntervalIndex(data, closed=closed)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["right", "left", "both", "neither"])
def test_interval_index_empty(closed):
    pindex = pd.IntervalIndex([], closed=closed)
    gindex = IntervalIndex([], closed=closed)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["right", "left", "both", "neither"])
@pytest.mark.parametrize(
    "data",
    [
        ([pd.Interval(1, 6), pd.Interval(1, 10), pd.Interval(1, 3)]),
        (
            [
                pd.Interval(3.5, 6.0),
                pd.Interval(1.0, 7.0),
                pd.Interval(0.0, 10.0),
            ]
        ),
        (
            [
                pd.Interval(50, 100, closed="left"),
                pd.Interval(1.0, 7.0, closed="left"),
                pd.Interval(16, 322, closed="left"),
            ]
        ),
        (
            [
                pd.Interval(50, 100, closed="right"),
                pd.Interval(1.0, 7.0, closed="right"),
                pd.Interval(16, 322, closed="right"),
            ]
        ),
    ],
)
def test_interval_index_many_params(data, closed):
    pindex = pd.IntervalIndex(data, closed=closed)
    gindex = IntervalIndex(data, closed=closed)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
def test_interval_index_from_breaks(closed):
    breaks = [0, 3, 6, 10]
    pindex = pd.IntervalIndex.from_breaks(breaks, closed=closed)
    gindex = IntervalIndex.from_breaks(breaks, closed=closed)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize(
    "start, stop, freq, periods",
    [
        (0.0, None, 0.2, 5),
        (0.0, 1.0, None, 5),
        pytest.param(
            0.0,
            1.0,
            0.2,
            None,
            marks=pytest.mark.xfail(
                condition=not PANDAS_GE_210,
                reason="https://github.com/pandas-dev/pandas/pull/54477",
            ),
        ),
        (None, 1.0, 0.2, 5),
        pytest.param(
            0.0,
            1.0,
            0.1,
            None,
            marks=pytest.mark.xfail(
                condition=not PANDAS_GE_210,
                reason="https://github.com/pandas-dev/pandas/pull/54477",
            ),
        ),
        (0.0, 1.0, None, 10),
        (0.0, None, 0.25, 4),
        (1.0, None, 2.5, 2),
    ],
)
def test_interval_range_floating(start, stop, freq, periods):
    expected = pd.interval_range(
        start=start, end=stop, freq=freq, periods=periods
    )
    got = interval_range(start=start, end=stop, freq=freq, periods=periods)
    assert_eq(expected, got)


def test_intervalindex_empty_typed_non_int():
    data = np.array([], dtype="datetime64[ns]")
    result = cudf.IntervalIndex(data)
    expected = pd.IntervalIndex(data)
    assert_eq(result, expected)
