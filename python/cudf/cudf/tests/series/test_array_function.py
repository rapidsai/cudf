# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "func",
    [
        np.mean,
        np.sum,
        lambda x: np.var(x, ddof=1),
        np.unique,
        lambda x: np.dot(x, x),
        np.linalg.norm,
    ],
)
def test_array_func_cudf_series(func):
    np_ar = np.arange(10, dtype=np.float32)
    cudf_ser = cudf.Series(np_ar)
    expect = func(np_ar)
    got = func(cudf_ser)
    if np.isscalar(expect):
        assert expect == got
    else:
        assert_eq(cudf.Series(expect), got)


@pytest.mark.parametrize("index", [None, [1, 2, 3]])
def test_list_input_array_func(index):
    s = cudf.Series(np.array([1, 2, 3]), index=index)
    with pytest.raises(TypeError):
        np.concatenate([s, s, s])
