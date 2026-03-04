# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cudf


@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.cov(x, x),
        lambda x: np.dot(x, x),
        np.linalg.norm,
        np.linalg.det,
    ],
)
def test_array_func_missing_cudf_multi_index(func):
    levels = [["a", "b"], ["c", "d"]]
    codes = [[0, 1], [1, 0]]

    cudf_multi_index = cudf.MultiIndex(levels, codes)
    with pytest.raises(TypeError):
        func(cudf_multi_index)
