# Copyright (c) 2022, NVIDIA CORPORATION.

import cupy as cp

import cudf


def test_basic():
    df = cudf.DataFrame()
    assert df.spillable is True
    df = cudf.DataFrame({"a": [1, 2, 3]})
    assert df.spillable is True
    s = cudf.Series([1, 2])
    assert s.spillable is True
    idx = cudf.Index([1, 2, 3])
    assert idx.spillable is True
    idx = cudf.Index(range(3))
    assert idx.spillable is False


def test_groupby():
    df = cudf.DataFrame(
        {
            "a": [
                1,
                1,
            ]
        }
    )
    gb = df.groupby("a")
    assert df.spillable is False
    del gb
    assert df.spillable is True


def test_cupy():
    s = cudf.Series([1, 2, 3])
    _ = cp.asarray(s)
    assert s.spillable is False
