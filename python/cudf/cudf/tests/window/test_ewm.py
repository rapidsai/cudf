# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [5.0, cudf.NA, 3.0, cudf.NA, 8.5],
        [5.0, cudf.NA, 3.0, cudf.NA, cudf.NA, 4.5],
        [5.0, cudf.NA, 3.0, 4.0, cudf.NA, 5.0],
    ],
)
@pytest.mark.parametrize(
    "params",
    [
        {"com": 0.1},
        {"com": 0.5},
        {"span": 1.5},
        {"span": 2.5},
        {"halflife": 0.5},
        {"halflife": 1.5},
        {"alpha": 0.1},
        {"alpha": 0.5},
    ],
)
@pytest.mark.parametrize("adjust", [True, False])
def test_ewma(data, params, adjust):
    """
    The most basic test asserts that we obtain
    the same numerical values as pandas for various
    sets of keyword arguemnts that effect the raw
    coefficients of the formula
    """
    gsr = cudf.Series(data, dtype="float64")
    psr = gsr.to_pandas()

    expect = psr.ewm(**params, adjust=adjust).mean()
    got = gsr.ewm(**params, adjust=adjust).mean()

    assert_eq(expect, got)


def test_ewm_leading_nulls():
    gsr = cudf.Series([None, 1.0, None, 1.0, 1.0], dtype="float64")
    psr = gsr.to_pandas()

    got = gsr.ewm(com=3, adjust=False, ignore_na=False, min_periods=0).mean()
    expect = psr.ewm(
        com=3, adjust=False, ignore_na=False, min_periods=0
    ).mean()
    assert_eq(expect, got)
