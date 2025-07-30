# Copyright (c) 2025, NVIDIA CORPORATION.


import cudf


def test_series_memory_usage():
    sr = cudf.Series([1, 2, 3, 4], dtype="int64")
    assert sr.memory_usage() == 32

    sliced_sr = sr[2:]
    assert sliced_sr.memory_usage() == 16

    sliced_sr[3] = None
    assert sliced_sr.memory_usage() == 80

    sr = cudf.Series(["hello world", "rapids ai", "abc", "z"])
    assert sr.memory_usage() == 44

    assert sr[3:].memory_usage() == 9  # z
    assert sr[:1].memory_usage() == 19  # hello world
