# Copyright (c) 2021-2025, NVIDIA CORPORATION.


import pytest

import cudf


def test_scalar_deprecation():
    with pytest.warns(FutureWarning):
        cudf.Scalar(1)
