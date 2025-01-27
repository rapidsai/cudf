# Copyright (c) 2023-2024, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

nsamps = 1000
reductions = ["sum", "min", "max", "mean", "var", "std"]


pytestmark = pytest.mark.assert_eq(fn=np.testing.assert_allclose)


@pytest.fixture(scope="module")
def sr():
    rng = np.random.default_rng(42)
    return pd.Series(rng.random(nsamps))


@pytest.mark.parametrize("op", reductions)
def test_numpy_series_reductions(sr, op):
    return getattr(np, op)(sr)


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(42)
    return pd.DataFrame({"A": rng.random(nsamps), "B": rng.random(nsamps)})


@pytest.mark.parametrize("op", reductions)
def test_numpy_dataframe_reductions(df, op):
    return getattr(np, op)(df)


def test_numpy_dot(df):
    return np.dot(df, df.T)


@pytest.mark.skip(
    reason="AttributeError: 'ndarray' object has no attribute '_fsproxy_wrapped'"
)
def test_numpy_fft(sr):
    fft = np.fft.fft(sr)
    return fft


def test_numpy_sort(df):
    return np.sort(df)


@pytest.mark.parametrize("percentile", [0, 25, 50, 75, 100])
def test_numpy_percentile(df, percentile):
    return np.percentile(df, percentile)


def test_numpy_unique(df):
    return np.unique(df)


def test_numpy_transpose(df):
    return np.transpose(df)
