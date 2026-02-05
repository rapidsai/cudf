# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.testing import assert_groupby_results_equal


@pytest.mark.parametrize("shift_perc", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("direction", [1, -1])
def test_groupby_diff_row(shift_perc, direction):
    nelem = 20
    pdf = pd.DataFrame(np.ones((nelem, 4)), columns=["x", "y", "val", "val2"])
    gdf = cudf.from_pandas(pdf)
    n_shift = int(nelem * shift_perc) * direction

    expected = pdf.groupby(["x", "y"]).diff(periods=n_shift)
    got = gdf.groupby(["x", "y"]).diff(periods=n_shift)

    assert_groupby_results_equal(expected, got)


@pytest.mark.parametrize("shift_perc", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("direction", [1, -1])
def test_groupby_diff_row_mixed_numerics(shift_perc, direction):
    nelem = 20
    rng = np.random.default_rng(0)

    int_key = rng.integers(0, 10, size=nelem)
    int_vals = rng.integers(0, 100, size=nelem)
    float_vals = rng.random(size=nelem).astype("float32")
    datetime_vals = pd.to_datetime(
        rng.integers(0, 10**12, size=nelem), unit="ns"
    )
    timedelta_vals = pd.to_timedelta(rng.integers(0, 10**9, size=nelem))

    null_masks = {col: rng.random(nelem) < 0.4 for col in ["1", "2", "3", "4"]}

    pdf = pd.DataFrame(
        {
            "0": int_key,
            "1": int_vals,
            "2": float_vals,
            "3": datetime_vals,
            "4": timedelta_vals,
        }
    )
    for col, mask in null_masks.items():
        pdf.loc[mask, col] = None

    gdf = cudf.from_pandas(pdf)
    pdf = gdf.to_pandas()
    n_shift = int(nelem * shift_perc) * direction

    expected = pdf.groupby(["0"]).diff(periods=n_shift)
    got = gdf.groupby(["0"]).diff(periods=n_shift)

    assert_groupby_results_equal(expected, got)


def test_groupby_diff_row_zero_shift():
    nelem = 20
    rng = np.random.default_rng(0)

    int_key = rng.integers(0, 10, size=nelem)
    int_vals = rng.integers(0, 100, size=nelem)
    float_vals = rng.random(size=nelem).astype("float32")
    datetime_vals = pd.to_datetime(
        rng.integers(0, 10**12, size=nelem), unit="ns"
    )
    timedelta_vals = pd.to_timedelta(rng.integers(0, 10**9, size=nelem))

    null_masks = {col: rng.random(nelem) < 0.4 for col in ["1", "2", "3", "4"]}

    pdf = pd.DataFrame(
        {
            "0": int_key,
            "1": int_vals,
            "2": float_vals,
            "3": datetime_vals,
            "4": timedelta_vals,
        }
    )
    for col, mask in null_masks.items():
        pdf.loc[mask, col] = None

    gdf = cudf.from_pandas(pdf)

    expected = gdf
    got = gdf.groupby(["0"]).shift(periods=0)

    assert_groupby_results_equal(
        expected[["1", "2", "3", "4"]], got[["1", "2", "3", "4"]]
    )


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
def test_groupby_fillna_multi_value():
    nelem = 20
    rng = np.random.default_rng(0)

    int_key = rng.integers(0, 10, size=nelem)
    int_vals = rng.integers(0, 100, size=nelem)
    float_vals = rng.random(size=nelem).astype("float32")
    datetime_vals = pd.to_datetime(
        rng.integers(0, 10**12, size=nelem), unit="ms"
    )
    timedelta_vals = pd.to_timedelta(rng.integers(0, 10**9, size=nelem))
    decimal_vals = rng.integers(0, 1000, size=nelem)
    str_vals = [f"str_{rng.integers(0, 100)}" for _ in range(nelem)]

    null_masks = {
        col: rng.random(nelem) < 0.4 for col in ["1", "2", "3", "4", "5", "6"]
    }

    pdf = pd.DataFrame(
        {
            "0": int_key,
            "1": int_vals,
            "2": float_vals,
            "3": datetime_vals,
            "4": timedelta_vals,
            "5": decimal_vals,
            "6": str_vals,
        }
    )
    for col, mask in null_masks.items():
        pdf.loc[mask, col] = None

    gdf = cudf.from_pandas(pdf)
    gdf["5"] = gdf["5"].astype(cudf.Decimal64Dtype(precision=10, scale=2))

    key_col = "0"
    value_cols = ["1", "2", "3", "4", "5", "6"]
    pdf = gdf.to_pandas()

    fill_values = {
        name: pdf[name].loc[pdf[name].first_valid_index()]
        for name in value_cols
    }
    fill_values["4"] = fill_values["4"].to_numpy()
    with pytest.warns(FutureWarning):
        expect = pdf.groupby(key_col).fillna(value=fill_values)
    with pytest.warns(FutureWarning):
        got = gdf.groupby(key_col).fillna(value=fill_values)

    assert_groupby_results_equal(expect[value_cols], got[value_cols])


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
def test_groupby_fillna_multi_value_df():
    nelem = 20
    rng = np.random.default_rng(0)

    int_key = rng.integers(0, 10, size=nelem)
    int_vals = rng.integers(0, 100, size=nelem)
    float_vals = rng.random(size=nelem).astype("float32")
    datetime_vals = pd.to_datetime(
        rng.integers(0, 10**12, size=nelem), unit="ms"
    )
    timedelta_vals = pd.to_timedelta(rng.integers(0, 10**9, size=nelem))
    str_vals = [f"str_{rng.integers(0, 100)}" for _ in range(nelem)]

    null_masks = {
        col: rng.random(nelem) < 0.4 for col in ["1", "2", "3", "4", "5"]
    }

    pdf = pd.DataFrame(
        {
            "0": int_key,
            "1": int_vals,
            "2": float_vals,
            "3": datetime_vals,
            "4": timedelta_vals,
            "5": str_vals,
        }
    )
    for col, mask in null_masks.items():
        pdf.loc[mask, col] = None

    gdf = cudf.from_pandas(pdf)

    key_col = "0"
    value_cols = ["1", "2", "3", "4", "5"]
    pdf = gdf.to_pandas()

    fill_values = {
        name: pdf[name].loc[pdf[name].first_valid_index()]
        for name in value_cols
    }
    fill_values["4"] = fill_values["4"].to_numpy()
    fill_values = pd.DataFrame(fill_values, index=pdf.index)
    with pytest.warns(FutureWarning):
        expect = pdf.groupby(key_col).fillna(value=fill_values)

    fill_values = cudf.from_pandas(fill_values)
    with pytest.warns(FutureWarning):
        got = gdf.groupby(key_col).fillna(value=fill_values)

    assert_groupby_results_equal(expect[value_cols], got[value_cols])


def test_groupby_select_then_diff():
    pdf = pd.DataFrame(
        {"a": [1, 1, 1, 2, 2], "b": [1, 2, 3, 4, 5], "c": [3, 4, 5, 6, 7]}
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby("a")["c"].diff(1)
    actual = gdf.groupby("a")["c"].diff(1)

    assert_groupby_results_equal(expected, actual)
