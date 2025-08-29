# Copyright (c) 2025, NVIDIA CORPORATION.


import cudf


def test_list_struct_list_memory_usage():
    df = cudf.DataFrame({"a": [[{"b": [1]}]]})
    assert df.memory_usage().sum() == 16
