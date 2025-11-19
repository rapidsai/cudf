# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import datetime

import pandas as pd
import pyarrow as pa
import pytest

import cudf


@pytest.mark.parametrize(
    "scalar",
    [
        1,
        1.0,
        "a",
        datetime.datetime(2020, 1, 1),
        datetime.timedelta(1),
        pd.Interval(1, 2),
    ],
)
def test_index_to_pandas_arrow_type_nullable_raises(scalar):
    pa_array = [scalar, None]
    midx = cudf.MultiIndex(levels=[pa_array], codes=[[0]])
    with pytest.raises(ValueError):
        midx.to_pandas(nullable=True, arrow_type=True)


@pytest.mark.parametrize(
    "scalar",
    [1, 1.0, "a", datetime.datetime(2020, 1, 1), datetime.timedelta(1)],
)
def test_index_to_pandas_arrow_type(scalar):
    pa_array = pa.array([scalar, None])
    midx = cudf.MultiIndex(levels=[pa_array], codes=[[0]])
    result = midx.to_pandas(arrow_type=True)
    expected = pd.MultiIndex(
        levels=[pd.arrays.ArrowExtensionArray(pa_array)], codes=[[0]]
    )
    pd.testing.assert_index_equal(result, expected)
