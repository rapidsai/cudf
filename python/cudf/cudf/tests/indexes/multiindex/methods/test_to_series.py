# Copyright (c) 2025, NVIDIA CORPORATION.

import pytest

import cudf


def test_multiindex_to_series_error():
    midx = cudf.MultiIndex.from_tuples([("a", "b")])
    with pytest.raises(NotImplementedError):
        midx.to_series()
