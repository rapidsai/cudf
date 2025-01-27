# Copyright (c) 2018-2024, NVIDIA CORPORATION.

import numpy as np

from cudf import Series


def test_to_dense_array():
    rng = np.random.default_rng(seed=0)
    data = rng.random(8)
    mask = np.asarray([0b11010110]).astype(np.byte)

    sr = Series.from_masked_array(data=data, mask=mask, null_count=3)
    assert sr.has_nulls
    assert sr.null_count != len(sr)
    filled = sr.to_numpy(na_value=np.nan)
    dense = sr.dropna().to_numpy()
    assert dense.size < filled.size
    assert filled.size == len(sr)
