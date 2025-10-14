# Copyright (c) 2025, NVIDIA CORPORATION.

import cudf
from cudf.testing import assert_eq


def test_struct_of_struct_copy():
    sr = cudf.Series([{"a": {"b": 1}}])
    assert_eq(sr, sr.copy())
