# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_to_pandas_regular_range_matches_pandas():
    # Round-trip through cudf must preserve the same DatetimeIndex (incl. freq)
    # that pandas would produce on its own.
    pidx = pd.date_range("2001-01-01", periods=5, freq="D")
    gidx = cudf.from_pandas(pidx)

    actual = gidx.to_pandas()
    assert actual.freq == pidx.freq
    assert_eq(actual, pidx)


def test_to_pandas_after_take_matches_pandas():
    # Regression: prior to the fix, to_pandas() raised. Now it should
    # match pandas's own take() output.
    pidx = pd.date_range("2001-01-01", periods=11, freq="D")
    gidx = cudf.from_pandas(pidx)

    perm = [4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6]
    expected = pidx.take(perm)
    actual = gidx.take(perm).to_pandas()

    assert actual.freq is None
    assert expected.freq is None
    assert_eq(actual, expected)


def test_to_pandas_arrow_type_values_match_pandas():
    pidx = pd.date_range("2001-01-01", periods=5, freq="D")
    gidx = cudf.from_pandas(pidx)

    actual = gidx.to_pandas(arrow_type=True)
    # arrow-backed Index has no freq — compare the underlying values.
    assert_eq(actual.to_numpy(), pidx.to_numpy())


def test_to_pandas_externally_set_stale_freq_matches_pandas_inferred():
    # If a stale freq sneaks in (e.g. via deserialization or external mutation),
    # to_pandas() should produce the freq pandas would have inferred from the
    # values, rather than raising.
    pidx = pd.date_range("2001-01-01", periods=5, freq="D")
    gidx = cudf.from_pandas(pidx)
    gidx._freq = cudf.DateOffset._from_freqstr(
        "h"
    )  # nonsense for daily values

    actual = gidx.to_pandas()
    expected_freq = pidx.inferred_freq  # what pandas would infer

    assert actual.freq == pd.tseries.frequencies.to_offset(expected_freq)
    assert_eq(actual.values, pidx.values)
