# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf
from cudf.core._compat import PANDAS_CURRENT_SUPPORTED_VERSION, PANDAS_VERSION
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize(
    "data",
    [
        # basics
        {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]},
        {"A": [1.0, None, 3.0], "B": [4.0, None, 6.0]},
        {"A": [None, 2.0, 3.0], "B": [4.0, 5.0, None]},
    ],
)
def test_interpolate_dataframe(data):
    # Pandas interpolate methods do not seem to work
    # with nullable dtypes yet, so this method treats
    # NAs as NaNs
    # https://github.com/pandas-dev/pandas/issues/40252
    axis = 0
    method = "linear"
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    expect = pdf.interpolate(method=method, axis=axis)
    got = gdf.interpolate(method=method, axis=axis)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data,kwargs",
    [
        (
            {"A": ["a", "b", "c"], "B": ["d", "e", "f"]},
            {"axis": 0, "method": "linear"},
        ),
        ({"A": [1, 2, 3]}, {"method": "pad", "limit_direction": "forward"}),
        ({"A": [1, 2, 3]}, {"method": "ffill", "limit_direction": "forward"}),
        ({"A": [1, 2, 3]}, {"method": "bfill", "limit_direction": "backward"}),
        (
            {"A": [1, 2, 3]},
            {"method": "backfill", "limit_direction": "backward"},
        ),
    ],
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Does not fail on older versions of pandas",
)
def test_interpolate_dataframe_error_cases(data, kwargs):
    gsr = cudf.DataFrame(data)
    psr = gsr.to_pandas()

    assert_exceptions_equal(
        lfunc=psr.interpolate,
        rfunc=gsr.interpolate,
        lfunc_args_and_kwargs=([], kwargs),
        rfunc_args_and_kwargs=([], kwargs),
    )
