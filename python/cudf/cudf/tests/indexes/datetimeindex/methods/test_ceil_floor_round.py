# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "resolution", ["D", "h", "min", "s", "ms", "us", "ns"]
)
@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_index_datetime_ceil(resolution, method):
    cuidx = cudf.DatetimeIndex([1000000, 2000000, 3000000, 4000000, 5000000])
    pidx = cuidx.to_pandas()

    expected = getattr(pidx, method)(resolution)
    result = getattr(cuidx, method)(resolution)

    assert_eq(expected, result)
