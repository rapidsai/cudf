# Copyright (c) 2025, NVIDIA CORPORATION.


import cudf


def test_rangeindex_slice_attr_name():
    start, stop = 0, 10
    rg = cudf.RangeIndex(start, stop, name="myindex")
    sliced_rg = rg[0:9]
    assert rg.name == sliced_rg.name
