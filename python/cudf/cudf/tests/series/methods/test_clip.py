# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    ("data", "lower", "upper"),
    [
        ([1, 2, 3, 4, 5], 2, 4),
        ([1, 2, 3, 4, 5], 2, None),
        ([1, 2, 3, 4, 5], None, 4),
        ([1, 2, 3, 4, 5], None, None),
        ([1, 2, 3, 4, 5], 4, 2),
        ([1.0, 2.0, 3.0, 4.0, 5.0], 4, 2),
        (pd.Series([1, 2, 3, 4, 5], dtype="int32"), 4, 2),
        (["a", "b", "c", "d", "e"], "b", "d"),
        (["a", "b", "c", "d", "e"], "b", None),
        (["a", "b", "c", "d", "e"], None, "d"),
        (["a", "b", "c", "d", "e"], "d", "b"),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_series_clip(data, lower, upper, inplace):
    psr = pd.Series(data)
    gsr = cudf.from_pandas(psr)

    expect = psr.clip(lower=lower, upper=upper)
    got = gsr.clip(lower=lower, upper=upper, inplace=inplace)

    if inplace is True:
        assert_eq(expect, gsr)
    else:
        assert_eq(expect, got)


def test_series_exceptions_for_clip():
    with pytest.raises(ValueError):
        cudf.Series([1, 2, 3, 4]).clip([1, 2], [2, 3])

    with pytest.raises(NotImplementedError):
        cudf.Series([1, 2, 3, 4]).clip(1, 2, axis=0)
