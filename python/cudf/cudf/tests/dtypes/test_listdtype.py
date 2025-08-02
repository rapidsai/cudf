# Copyright (c) 2025, NVIDIA CORPORATION.


import cudf


def test_listdtype_hash():
    a = cudf.ListDtype("int64")
    b = cudf.ListDtype("int64")

    assert hash(a) == hash(b)

    c = cudf.ListDtype("int32")

    assert hash(a) != hash(c)
