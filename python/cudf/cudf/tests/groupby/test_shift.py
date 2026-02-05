# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq, assert_groupby_results_equal


@pytest.mark.parametrize("shift_perc", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("fill_value", [None, np.nan, 42])
def test_groupby_shift_row(shift_perc, direction, fill_value):
    nelem = 20
    pdf = pd.DataFrame(np.ones((nelem, 3)), columns=["x", "y", "val"])
    gdf = cudf.from_pandas(pdf)
    n_shift = int(nelem * shift_perc) * direction

    expected = pdf.groupby(["x", "y"]).shift(
        periods=n_shift, fill_value=fill_value
    )
    got = gdf.groupby(["x", "y"]).shift(periods=n_shift, fill_value=fill_value)

    assert_groupby_results_equal(expected, got)


@pytest.mark.parametrize("shift_perc", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize(
    "fill_value",
    [
        None,
        pytest.param(
            0,
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/10608"
            ),
        ),
        pytest.param(
            42,
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/10608"
            ),
        ),
    ],
)
def test_groupby_shift_row_mixed_numerics(shift_perc, direction, fill_value):
    nelem = 20
    rng = np.random.default_rng(0)

    int_key = rng.integers(0, 10, size=nelem)
    int_vals = rng.integers(0, 100, size=nelem)
    float_vals = rng.random(size=nelem).astype("float32")
    datetime_vals = pd.to_datetime(rng.integers(0, 10**12, size=nelem))
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

    expected = pdf.groupby(["0"]).shift(periods=n_shift, fill_value=fill_value)
    got = gdf.groupby(["0"]).shift(periods=n_shift, fill_value=fill_value)

    assert_groupby_results_equal(expected, got)


# TODO: Shifting list columns is currently unsupported because we cannot
# construct a null list scalar in python. Support once it is added.
@pytest.mark.parametrize("shift_perc", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("direction", [1, -1])
def test_groupby_shift_row_mixed(shift_perc, direction):
    nelem = 20
    rng = np.random.default_rng(0)

    int_key = rng.integers(0, 10, size=nelem)
    int_vals = rng.integers(0, 100, size=nelem)
    str_vals = [f"str_{rng.integers(0, 100)}" for _ in range(nelem)]
    datetime_vals = pd.to_datetime(rng.integers(0, 10**12, size=nelem))
    timedelta_vals = pd.to_timedelta(rng.integers(0, 10**9, size=nelem))

    null_masks = {col: rng.random(nelem) < 0.4 for col in ["1", "2", "3", "4"]}

    pdf = pd.DataFrame(
        {
            "0": int_key,
            "1": int_vals,
            "2": str_vals,
            "3": datetime_vals,
            "4": timedelta_vals,
        }
    )
    for col, mask in null_masks.items():
        pdf.loc[mask, col] = None

    gdf = cudf.from_pandas(pdf)
    pdf = gdf.to_pandas()
    n_shift = int(nelem * shift_perc) * direction

    expected = pdf.groupby(["0"]).shift(periods=n_shift)
    got = gdf.groupby(["0"]).shift(periods=n_shift)

    assert_groupby_results_equal(expected, got)


@pytest.mark.parametrize("shift_perc", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize(
    "fill_value",
    [
        [
            42,
            "fill",
            np.datetime64(123, "ns"),
            np.timedelta64(456, "ns"),
        ]
    ],
)
def test_groupby_shift_row_mixed_fill(shift_perc, direction, fill_value):
    nelem = 20
    rng = np.random.default_rng(0)

    int_key = rng.integers(0, 10, size=nelem)
    int_vals = rng.integers(0, 100, size=nelem)
    str_vals = [f"str_{rng.integers(0, 100)}" for _ in range(nelem)]
    datetime_vals = pd.to_datetime(rng.integers(0, 10**12, size=nelem))
    timedelta_vals = pd.to_timedelta(rng.integers(0, 10**9, size=nelem))

    null_masks = {col: rng.random(nelem) < 0.4 for col in ["1", "2", "3", "4"]}

    pdf = pd.DataFrame(
        {
            "0": int_key,
            "1": int_vals,
            "2": str_vals,
            "3": datetime_vals,
            "4": timedelta_vals,
        }
    )
    for col, mask in null_masks.items():
        pdf.loc[mask, col] = None

    gdf = cudf.from_pandas(pdf)
    pdf = gdf.to_pandas()
    n_shift = int(nelem * shift_perc) * direction

    # Pandas does not support specifying different fill_value by column, so we
    # simulate it column by column
    expected = pdf.copy()
    for col, single_fill in zip(pdf.iloc[:, 1:], fill_value, strict=True):
        expected[col] = (
            pdf[col]
            .groupby(pdf["0"])
            .shift(periods=n_shift, fill_value=single_fill)
        )

    got = gdf.groupby(["0"]).shift(periods=n_shift, fill_value=fill_value)

    assert_groupby_results_equal(
        expected[["1", "2", "3", "4"]], got[["1", "2", "3", "4"]]
    )


@pytest.mark.parametrize("fill_value", [None, 0, 42])
def test_groupby_shift_row_zero_shift(fill_value):
    nelem = 20
    rng = np.random.default_rng(0)

    int_key = rng.integers(0, 10, size=nelem)
    int_vals = rng.integers(0, 100, size=nelem)
    float_vals = rng.random(size=nelem).astype("float32")
    datetime_vals = pd.to_datetime(rng.integers(0, 10**12, size=nelem))
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
    got = gdf.groupby(["0"]).shift(periods=0, fill_value=fill_value)

    assert_groupby_results_equal(
        expected[["1", "2", "3", "4"]], got[["1", "2", "3", "4"]]
    )


def test_groupby_select_then_shift():
    pdf = pd.DataFrame(
        {"a": [1, 1, 1, 2, 2], "b": [1, 2, 3, 4, 5], "c": [3, 4, 5, 6, 7]}
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby("a")["c"].shift(1)
    actual = gdf.groupby("a")["c"].shift(1)

    assert_groupby_results_equal(expected, actual)


def test_groupby_shift_series_multiindex():
    idx = cudf.MultiIndex.from_tuples(
        [("a", 1), ("a", 2), ("b", 1), ("b", 2)], names=["f", "s"]
    )
    ser = cudf.Series(range(4), index=idx)
    result = ser.groupby(level=0).shift(1)
    expected = ser.to_pandas().groupby(level=0).shift(1)
    assert_eq(expected, result)
