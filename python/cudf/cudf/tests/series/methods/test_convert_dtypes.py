# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data, dtype",
    [
        ([1, 2, 3], "int8"),
        ([1, 2, 3], "int64"),
        ([1.1, 2.2, 3.3], "float32"),
        ([1.0, 2.0, 3.0], "float32"),
        ([1.0, 2.0, 3.0], "float64"),
        (["a", "b", "c"], "str"),
        (["a", "b", "c"], "category"),
        (["2001-01-01", "2001-01-02", "2001-01-03"], "datetime64[ns]"),
    ],
)
def test_convert_dtypes(data, dtype):
    s = pd.Series(data, dtype=dtype)
    gs = cudf.Series(data, dtype=dtype)
    expect = s.convert_dtypes()

    # because we don't have distinct nullable types, we check that we
    # get the same result if we convert to nullable pandas types:
    nullable = dtype not in ("category", "datetime64[ns]")
    got = gs.convert_dtypes().to_pandas(nullable=nullable)
    assert_eq(expect, got)


def test_convert_integer_false_convert_floating_true():
    data = [1.000000000000000000000000001, 1]
    expected = pd.Series(data).convert_dtypes(
        convert_integer=False, convert_floating=True
    )
    result = (
        cudf.Series(data)
        .convert_dtypes(convert_integer=False, convert_floating=True)
        .to_pandas(nullable=True)
    )
    assert_eq(result, expected)
