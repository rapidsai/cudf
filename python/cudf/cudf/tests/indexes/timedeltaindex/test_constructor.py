# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_timedelta_constructor():
    data = [43534, 43543, 37897, 2000]
    dtype = "timedelta64[ns]"
    expected = pd.TimedeltaIndex(data=data, dtype=dtype)
    actual = cudf.TimedeltaIndex(data=data, dtype=dtype)

    assert_eq(expected, actual)

    expected = pd.TimedeltaIndex(data=pd.Series(data), dtype=dtype)
    actual = cudf.TimedeltaIndex(data=cudf.Series(data), dtype=dtype)

    assert_eq(expected, actual)


@pytest.mark.parametrize("name", [None, "delta-index"])
def test_create_TimedeltaIndex(timedelta_types_as_str, name):
    gdi = cudf.TimedeltaIndex(
        [1132223, 2023232, 342234324, 4234324],
        dtype=timedelta_types_as_str,
        name=name,
    )
    pdi = gdi.to_pandas()
    assert_eq(pdi, gdi)
