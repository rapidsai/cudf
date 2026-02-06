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
from cudf.testing._utils import assert_exceptions_equal


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


# TODO: cudf.fillna does not support decimal column to column fill yet
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


@pytest.mark.parametrize(
    "by",
    [pd.Series([1, 1, 2, 2, 3, 4]), lambda x: x % 2 == 0, pd.Grouper(level=0)],
)
@pytest.mark.parametrize(
    "data", [[1, None, 2, None, 3, None], [1, 2, 3, 4, 5, 6]]
)
@pytest.mark.parametrize("args", [{"value": 42}, {"method": "ffill"}])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
def test_groupby_various_by_fillna(by, data, args):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    with pytest.warns(FutureWarning):
        expect = ps.groupby(by).fillna(**args)
    if isinstance(by, pd.Grouper):
        by = cudf.Grouper(level=by.level)
    with pytest.warns(FutureWarning):
        got = gs.groupby(by).fillna(**args)

    assert_groupby_results_equal(expect, got, check_dtype=False)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_groupby_fillna_method(method):
    nelem = 20
    rng = np.random.default_rng(0)

    int_key = rng.integers(0, 10, size=nelem)
    int_vals = rng.integers(0, 100, size=nelem)
    float_vals = rng.random(size=nelem).astype("float32")
    datetime_vals = pd.to_datetime(rng.integers(0, 10**12, size=nelem))
    timedelta_vals = pd.to_timedelta(rng.integers(0, 10**9, size=nelem))
    list_vals = [
        [rng.integers(0, 100) for _ in range(rng.integers(1, 5))]
        for _ in range(nelem)
    ]
    cat_vals = pd.Categorical(rng.choice(["a", "b", "c", "d"], size=nelem))
    decimal_vals = rng.integers(0, 1000, size=nelem)
    str_vals = [f"str_{rng.integers(0, 100)}" for _ in range(nelem)]

    null_masks = {
        col: rng.random(nelem) < 0.4
        for col in ["1", "2", "3", "4", "5", "6", "7", "8"]
    }

    pdf = pd.DataFrame(
        {
            "0": int_key,
            "1": int_vals,
            "2": float_vals,
            "3": datetime_vals,
            "4": timedelta_vals,
            "6": cat_vals,
            "7": decimal_vals,
            "8": str_vals,
        }
    )
    for col in ["1", "2", "3", "4", "6", "7", "8"]:
        pdf.loc[null_masks[col], col] = None

    gdf = cudf.from_pandas(pdf)

    list_series = cudf.Series(list_vals)
    list_series[null_masks["5"]] = None
    gdf.insert(5, "5", list_series)

    gdf["7"] = gdf["7"].astype(cudf.Decimal64Dtype(precision=10, scale=2))

    key_col = "0"
    value_cols = ["1", "2", "3", "4", "5", "6", "7", "8"]
    pdf = gdf.to_pandas()

    with pytest.warns(FutureWarning):
        expect = pdf.groupby(key_col).fillna(method=method)
    with pytest.warns(FutureWarning):
        got = gdf.groupby(key_col).fillna(method=method)

    assert_groupby_results_equal(
        expect[value_cols], got[value_cols], sort=False
    )


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Does not warn on older versions of pandas",
)
def test_cat_groupby_fillna():
    ps = pd.Series(["a", "b", "c"], dtype="category")
    gs = cudf.from_pandas(ps)

    with pytest.warns(FutureWarning):
        pg = ps.groupby(ps)
    gg = gs.groupby(gs)

    assert_exceptions_equal(
        lfunc=pg.fillna,
        rfunc=gg.fillna,
        lfunc_args_and_kwargs=(("d",), {}),
        rfunc_args_and_kwargs=(("d",), {}),
    )
