# Copyright (c) 2025, NVIDIA CORPORATION.


import numpy as np

import cudf


def test_index_comparision():
    start, stop = 10, 34
    rg = cudf.RangeIndex(start, stop)
    gi = cudf.Index(np.arange(start, stop))
    assert rg.equals(gi)
    assert gi.equals(rg)
    assert not rg[:-1].equals(gi)
    assert rg[:-1].equals(gi[:-1])
