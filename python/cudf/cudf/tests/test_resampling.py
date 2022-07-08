# Copyright (c) 2021-2022, NVIDIA CORPORATION.

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


@pytest.mark.parametrize("ts_resolution", ["ns", "s", "ms"])
def test_series_downsample_simple(ts_resolution):
    # Series with and index of 5min intervals:

    index = pd.date_range(start="2001-01-01", periods=10, freq="1T")
    psr = pd.Series(range(10), index=index)
    gsr = cudf.from_pandas(psr)
    gsr.index = gsr.index.astype(f"datetime64[{ts_resolution}]")
    assert_resample_results_equal(
        psr.resample("3T").sum(),
        gsr.resample("3T").sum(),
    )


def test_series_upsample_simple():
    # Series with and index of 5min intervals:

    index = pd.date_range(start="2001-01-01", periods=10, freq="1T")
    psr = pd.Series(range(10), index=index)
    gsr = cudf.from_pandas(psr)
    assert_resample_results_equal(
        psr.resample("3T").sum(),
        gsr.resample("3T").sum(),
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


def test_dataframe_resample_aggregation_simple():
    pdf = pd.DataFrame(
        np.random.randn(1000, 3),
        index=pd.date_range("1/1/2012", freq="S", periods=1000),
        columns=["A", "B", "C"],
    )
    gdf = cudf.from_pandas(pdf)
    assert_resample_results_equal(
        pdf.resample("3T").mean(), gdf.resample("3T").mean()
    )


def test_dataframe_resample_multiagg():
    pdf = pd.DataFrame(
        np.random.randn(1000, 3),
        index=pd.date_range("1/1/2012", freq="S", periods=1000),
        columns=["A", "B", "C"],
    )
    gdf = cudf.from_pandas(pdf)
    assert_resample_results_equal(
        pdf.resample("3T").agg(["sum", "mean", "std"]),
        gdf.resample("3T").agg(["sum", "mean", "std"]),
    )


def test_dataframe_resample_on():
    # test resampling on a specified column
    pdf = pd.DataFrame(
        {
            "x": np.random.randn(1000),
            "y": pd.date_range("1/1/2012", freq="S", periods=1000),
        }
    )
    gdf = cudf.from_pandas(pdf)
    assert_resample_results_equal(
        pdf.resample("3T", on="y").mean(), gdf.resample("3T", on="y").mean()
    )


def test_dataframe_resample_level():
    # test resampling on a specific level of a MultIndex
    pdf = pd.DataFrame(
        {
            "x": np.random.randn(1000),
            "y": pd.date_range("1/1/2012", freq="S", periods=1000),
        }
    )
    pdi = pd.MultiIndex.from_frame(pdf)
    pdf = pd.DataFrame({"a": np.random.randn(1000)}, index=pdi)
    gdf = cudf.from_pandas(pdf)
    assert_resample_results_equal(
        pdf.resample("3T", level="y").mean(),
        gdf.resample("3T", level="y").mean(),
    )


@pytest.mark.parametrize(
    "in_freq, sampling_freq, out_freq",
    [
        ("1ns", "1us", "us"),
        ("1us", "10us", "us"),
        ("ms", "100us", "us"),
        ("ms", "1s", "s"),
        ("s", "1T", "s"),
        ("1T", "30s", "s"),
        ("1D", "10D", "s"),
        ("10D", "1D", "s"),
    ],
)
def test_resampling_frequency_conversion(in_freq, sampling_freq, out_freq):
    # test that we cast to the appropriate frequency
    # when resampling:
    pdf = pd.DataFrame(
        {
            "x": np.random.randn(100),
            "y": pd.date_range("1/1/2012", freq=in_freq, periods=100),
        }
    )
    gdf = cudf.from_pandas(pdf)
    expect = pdf.resample(sampling_freq, on="y").mean()
    got = gdf.resample(sampling_freq, on="y").mean()
    assert_resample_results_equal(expect, got)

    assert got.index.dtype == np.dtype(f"datetime64[{out_freq}]")
