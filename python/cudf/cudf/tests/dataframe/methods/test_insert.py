# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_insert_reset_label_dtype():
    result = cudf.DataFrame({1: [2]})
    expected = pd.DataFrame({1: [2]})
    result.insert(1, "a", [2])
    expected.insert(1, "a", [2])
    assert_eq(result, expected)
