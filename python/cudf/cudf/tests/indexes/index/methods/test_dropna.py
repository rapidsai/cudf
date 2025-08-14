# Copyright (c) 2025, NVIDIA CORPORATION.

import pytest

import cudf


def test_dropna_bad_how():
    with pytest.raises(ValueError):
        cudf.Index([1]).dropna(how="foo")
