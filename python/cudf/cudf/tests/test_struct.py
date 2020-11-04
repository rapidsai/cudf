# Copyright (c) 2020, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [{}],
        [{"a": None}],
        [{"a": 1}],
        [{"a": "one"}],
        [{"a": 1}, {"a": 2}],
        [{"a": 1, "b": "one"}, {"a": 2, "b": "two"}],
        [{"b": "two", "a": None}, None, {"a": "one", "b": "two"}],
    ],
)
def test_create_struct_series(data):
    expect = pd.Series(data)
    got = cudf.Series(data)
    assert_eq(expect, got, check_dtype=False)
