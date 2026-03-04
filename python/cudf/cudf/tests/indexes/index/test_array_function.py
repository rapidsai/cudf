# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


def test_np_unique_cudf_index():
    np_ar = np.array([1, 1, 3])
    cudf_index = cudf.Index(np_ar)
    expect = cudf.Index(np.unique(np_ar))
    got = np.unique(cudf_index)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.cov(x, x),
        np.linalg.norm,
        np.linalg.det,
    ],
)
def test_array_func_missing_cudf_index(func):
    cudf_index = cudf.Index([1, 2, 3])
    with pytest.raises(TypeError):
        func(cudf_index)
