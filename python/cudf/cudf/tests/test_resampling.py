import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing._utils import assert_eq


def assert_resample_results_equal(lhs, rhs, **kwargs):
    assert_eq(
        lhs.sort_index(),
        rhs.sort_index(),
        check_dtype=False,
        check_freq=False,
        **kwargs,
    )


def test_series_downsample_simple():
    # Series with and index of 5min intervals:

    index = pd.date_range(start="2001-01-01", periods=10, freq="1T")
    psr = pd.Series(range(10), index=index)
    gsr = cudf.from_pandas(psr)
    assert_resample_results_equal(
        psr.resample("3T").sum(), gsr.resample("3T").sum(),
    )


def test_series_upsample_simple():
    # Series with and index of 5min intervals:

    index = pd.date_range(start="2001-01-01", periods=10, freq="1T")
    psr = pd.Series(range(10), index=index)
    gsr = cudf.from_pandas(psr)
    assert_resample_results_equal(
        psr.resample("3T").sum(), gsr.resample("3T").sum(),
    )


@pytest.mark.parametrize("rule", ["2S", "10S"])
def test_series_resample_ffill(rule):
    rng = pd.date_range("1/1/2012", periods=10, freq="5S")
    ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
    gts = cudf.from_pandas(ts)
    assert_resample_results_equal(
        ts.resample(rule).ffill(), gts.resample(rule).ffill()
    )


@pytest.mark.parametrize("rule", ["2S", "10S"])
def test_series_resample_bfill(rule):
    rng = pd.date_range("1/1/2012", periods=10, freq="5S")
    ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
    gts = cudf.from_pandas(ts)
    assert_resample_results_equal(
        ts.resample(rule).bfill(), gts.resample(rule).bfill()
    )


@pytest.mark.parametrize("rule", ["2S", "10S"])
def test_series_resample_asfreq(rule):
    rng = pd.date_range("1/1/2012", periods=100, freq="5S")
    ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
    gts = cudf.from_pandas(ts)
    assert_resample_results_equal(
        ts.resample(rule).asfreq(), gts.resample(rule).asfreq()
    )
