# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import PANDAS_CURRENT_SUPPORTED_VERSION, PANDAS_VERSION
from cudf.testing import assert_eq
from cudf.testing._utils import expect_warning_if


@pytest.mark.parametrize(
    "data1",
    [
        np.random.default_rng(seed=0).normal(-100, 100, 1000),
        np.random.default_rng(seed=0).integers(-50, 50, 1000),
        np.zeros(100),
        np.repeat(np.nan, 100),
        np.array([1.123, 2.343, np.nan, 0.0]),
        pa.array([5, 10, 53, None, np.nan, None]),
        pd.Series([1.1, 2.32, 43.4], index=[0, 4, 3]),
        np.array([], dtype="float64"),
        np.array([-3]),
    ],
)
@pytest.mark.parametrize(
    "data2",
    [
        np.random.default_rng(seed=0).normal(-100, 100, 1000),
        np.random.default_rng(seed=0).integers(-50, 50, 1000),
        np.zeros(100),
        np.repeat(np.nan, 100),
        np.array([1.123, 2.343, np.nan, 0.0]),
        pd.Series([1.1, 2.32, 43.4], index=[0, 500, 4000]),
        np.array([5]),
    ],
)
def test_cov1d(data1, data2):
    gs1 = cudf.Series(data1)
    gs2 = cudf.Series(data2)

    ps1 = gs1.to_pandas()
    ps2 = gs2.to_pandas()

    got = gs1.cov(gs2)
    ps1_align, ps2_align = ps1.align(ps2, join="inner")
    with expect_warning_if(
        (len(ps1_align.dropna()) == 1 and len(ps2_align.dropna()) > 0)
        or (len(ps2_align.dropna()) == 1 and len(ps1_align.dropna()) > 0),
        RuntimeWarning,
    ):
        expected = ps1.cov(ps2)
    np.testing.assert_approx_equal(got, expected, significant=8)


@pytest.mark.parametrize(
    "data1",
    [
        np.random.default_rng(seed=0).normal(-100, 100, 1000),
        np.random.default_rng(seed=0).integers(-50, 50, 1000),
        np.zeros(100),
        np.repeat(np.nan, 100),
        np.array([1.123, 2.343, np.nan, 0.0]),
        pa.array([5, 10, 53, None, np.nan, None]),
        pd.Series([1.1032, 2.32, 43.4], index=[0, 4, 3]),
        np.array([], dtype="float64"),
        np.array([-3]),
    ],
)
@pytest.mark.parametrize(
    "data2",
    [
        np.random.default_rng(seed=0).normal(-100, 100, 1000),
        np.random.default_rng(seed=0).integers(-50, 50, 1000),
        np.zeros(100),
        np.repeat(np.nan, 100),
        np.array([1.123, 2.343, np.nan, 0.0]),
        pd.Series([1.1, 2.32, 43.4], index=[0, 500, 4000]),
        np.array([5]),
    ],
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Warnings missing on older pandas (scipy version seems unrelated?)",
)
def test_corr1d(data1, data2, corr_method):
    if corr_method == "spearman":
        # Pandas uses scipy.stats.spearmanr code-path
        pytest.importorskip("scipy")

    gs1 = cudf.Series(data1)
    gs2 = cudf.Series(data2)

    ps1 = gs1.to_pandas()
    ps2 = gs2.to_pandas()

    got = gs1.corr(gs2, corr_method)

    ps1_align, ps2_align = ps1.align(ps2, join="inner")

    is_singular = (
        len(ps1_align.dropna()) == 1 and len(ps2_align.dropna()) > 0
    ) or (len(ps2_align.dropna()) == 1 and len(ps1_align.dropna()) > 0)
    is_identical = (
        len(ps1_align.dropna().unique()) == 1 and len(ps2_align.dropna()) > 0
    ) or (
        len(ps2_align.dropna().unique()) == 1 and len(ps1_align.dropna()) > 0
    )

    # Pearson correlation leads to division by 0 when either sample size is 1.
    # Spearman allows for size 1 samples, but will error if all data in a
    # sample is identical since the covariance is zero and so the correlation
    # coefficient is not defined.
    cond = ((is_singular or is_identical) and corr_method == "pearson") or (
        is_identical and not is_singular and corr_method == "spearman"
    )
    if corr_method == "spearman":
        # SciPy has shuffled around the warning it throws a couple of times.
        # It's not worth the effort of conditionally importing the appropriate
        # warning based on the scipy version, just catching a base Warning is
        # good enough validation.
        expected_warning = Warning
    elif corr_method == "pearson":
        expected_warning = RuntimeWarning

    with expect_warning_if(cond, expected_warning):
        expected = ps1.corr(ps2, corr_method)
    np.testing.assert_approx_equal(got, expected, significant=8)


@pytest.mark.parametrize(
    "data1",
    [
        [1, 2, 3, 4],
        [10, 1, 3, 5],
    ],
)
@pytest.mark.parametrize(
    "data2",
    [
        [1, 2, 3, 4],
        [10, 1, 3, 5],
    ],
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_cov_corr_datetime_timedelta(data1, data2, temporal_types_as_str):
    gsr1 = cudf.Series(data1, dtype=temporal_types_as_str)
    gsr2 = cudf.Series(data2, dtype=temporal_types_as_str)
    psr1 = gsr1.to_pandas()
    psr2 = gsr2.to_pandas()

    assert_eq(psr1.corr(psr2), gsr1.corr(gsr2))
    assert_eq(psr1.cov(psr2), gsr1.cov(gsr2))
