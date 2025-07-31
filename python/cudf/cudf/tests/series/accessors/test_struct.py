# Copyright (c) 2025, NVIDIA CORPORATION.

import pytest

import cudf


def test_struct_iterate_error():
    s = cudf.Series(
        [{"f2": {"a": "sf21"}, "f1": "a"}, {"f1": "sf12", "f2": None}]
    )
    with pytest.raises(TypeError):
        iter(s.struct)
