# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_rename_shallow_copy():
    idx = pd.Index([1])
    result = idx.rename("a")
    assert idx.to_numpy(copy=False) is result.to_numpy(copy=False)

    idx = cudf.Index([1])
    result = idx.rename("a")
    assert_eq(idx._column, result._column)
