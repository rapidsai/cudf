# Copyright (c) 2025, NVIDIA CORPORATION.


import cudf
from cudf.testing import assert_eq


def test_index_any():
    gidx = cudf.Index([1, 2, 3])
    pidx = gidx.to_pandas()

    assert_eq(pidx.any(), gidx.any())
