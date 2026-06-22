# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import string

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_nunique_all_null(dropna):
    data = [None, None]
    pd_ser = pd.Series(data)
    cudf_ser = cudf.Series(data)
    result = pd_ser.nunique(dropna=dropna)
    expected = cudf_ser.nunique(dropna=dropna)
    assert result == expected


def test_series_nunique():
    cd_s = cudf.Series([1, 3, 5, 7, 7])
    pd_s = cd_s.to_pandas()

    actual = cd_s.nunique()
    expected = pd_s.nunique()

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([], dtype="datetime64[ns]"),
        pd.Series(pd.date_range("2010-01-01", "2010-02-01")),
        pd.Series([None, None], dtype="datetime64[ns]"),
    ],
)
@pytest.mark.parametrize("nulls", ["none", "some"])
def test_datetime_nunique(data, nulls):
    psr = data.copy()
    rng = np.random.default_rng(seed=0)

    if len(data) > 0:
        if nulls == "some":
            p = rng.integers(0, len(data), 2)
            psr[p] = None

    gsr = cudf.from_pandas(psr)
    expected = psr.nunique()
    got = gsr.nunique()
    assert_eq(got, expected)


def test_categorical_nunique():
    nelem = 20
    rng = np.random.default_rng(seed=0)
    pd_cat = pd.Categorical(
        pd.Series(
            rng.choice(list(string.ascii_letters + string.digits), nelem),
            dtype="category",
        )
    )

    gser = cudf.Series(pd_cat)
    gdf_unique_count = gser.nunique()

    pser = pd.Series(pd_cat)
    pdf_unique = pser.unique()

    assert gdf_unique_count == len(pdf_unique)
